from typing import Sequence, Union, Any, Optional
from functools import reduce

import torch
import torch.nn as nn

from geotorch import orthogonal
from geotorch.parametrize import cached


cached.__doc__ = r"""Context-manager that enables the caching system (used for avoid recomputing rotation matrices)."""


def divide(numer, denom):
    """Numerically stable division."""
    epsilon = 1e-15
    return torch.sign(numer) * torch.sign(denom) * torch.exp(torch.log(numer.abs() + epsilon) - torch.log(denom.abs() + epsilon))


class VanillaMTL(nn.Module):
    def __init__(self, backbone, heads):
        super().__init__()
        self._backbone = [backbone]
        self.heads = heads
        self.rep = None
        self.grads = [None for _ in range(len(heads))]

    @property
    def backbone(self):
        return self._backbone[0]

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        self.backbone.train(mode)
        for head in self.heads:
            head.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super(VanillaMTL, self).to(*args, **kwargs)

    def _hook(self, index):
        def _hook_(g):
            self.grads[index] = g
        return _hook_

    def forward(self, x):
        preds = []
        out = self.backbone(x)

        if isinstance(out, (list, tuple)):
            rep, extra_out = out[0], out[1:]
            extra_out = list(extra_out)
        else:
            rep = out
            extra_out = []

        if self.training:
            self.rep = rep

        for i, head in enumerate(self.heads):
            rep_i = rep
            if self.training:
                rep_i = rep.detach().clone()
                rep_i.requires_grad = True
                rep_i.register_hook(self._hook(i))

            out_i = head(rep_i)
            if isinstance(out_i, (list, tuple)):
                preds.append(out_i[0])
                extra_out.append(out_i[1:])
            else:
                preds.append(out_i)

        if len(extra_out) == 0:
            return preds
        else:
            return preds, extra_out

    def backward(self, losses, backbone_loss=None, **kwargs):
        for loss in losses:
            loss.backward(**kwargs)

        if backbone_loss is not None:
            backbone_loss.backward(retain_graph=True)

        self.rep.backward(sum(self.grads))

    def mtl_parameters(self, recurse=True):
        return self.parameters(recurse=recurse)

    def model_parameters(self, recurse=True):
        for param in self.backbone.parameters(recurse=recurse):
            yield param

        for h in self.heads:
            for param in h.parameters(recurse=recurse):
                yield param


def rotate(points, rotation, total_size):
    if total_size != points.size(-2):
        points_lo, points_hi = points[..., :rotation.size(1), :], points[..., rotation.size(1):, :]
        point_lo = torch.einsum('ij,...jk->...ik', rotation, points_lo)
        return torch.cat((point_lo, points_hi), dim=-2)
    else:
        return torch.einsum('ij,...jk->...ik', rotation, points)


def rotate_back(points, rotation, total_size):
    return rotate(points, rotation.t(), total_size)


class RotateModule(nn.Module):
    def __init__(self, parent, item):
        super().__init__()

        self.parent = [parent]  # Dirty trick to not register parameters
        self.item = item

    def hook(self, g):
        self.p.grads[self.item] = g.clone()

    @property
    def p(self) -> 'RotateOnly':
        return self.parent[0]

    @property
    def R(self):
        return self.p.rotation[self.item]

    @property
    def weight(self):
        return self.p.weight[self.item] if hasattr(self.p, 'weight') else 1.

    def rotate(self, z):
        dim_post = -len(self.p.post_shape)
        dim_rot = -len(self.p.rotation_shape)
        og_shape = z.shape
        if dim_post == 0:
            z = z.unsqueeze(dim=-1)
            dim_post = -1

        z = z.flatten(start_dim=dim_post)
        z = z.flatten(start_dim=dim_rot - 1, end_dim=-2)

        return rotate(z, self.R.detach(), self.p.rotation_size).view(og_shape)

    def rotate_back(self, z):
        return rotate_back(z, self.R, self.p.rotation_size)

    def forward(self, z):
        new_z = self.rotate(z)
        if self.p.training:
            new_z.register_hook(self.hook)

        return new_z


class RotateOnly(nn.Module):
    r"""
    Implementation of the rotating part of RotoGrad as described in the original paper. [1]_

    The module takes as input a vector of shape ... x rotation_shape x

    Parameters
    ----------
    backbone
        Shared module.
    heads
        Task-specific modules.
    rotation_shape
        Shape of the shared representation to be rotated which, usually, is just the size of the backbone's output.
        Passing a shape is useful, for example, if you want to rotate an image with shape width x height.
    post_shape : optional, default=()
        Shape of the shared representation following the part to be rotated (if any). This part will be kept as it is.
        This is useful, for example, if you want to rotate only the channels of an image.
    normalize_losses : optional, default=False
        Whether to use this normalized losses to back-propagate through the task-specific parameters as well.
    burn_in_period : optional, default=20
        When back-propagating towards the shared parameters, *each task loss is normalized dividing by its initial
        value*, :math:`{L_k(t)}/{L_k(t_0 = 0)}`. This parameter sets a number of iterations after which the denominator
        will be replaced by the value of the loss at that iteration, that is, :math:`t_0 = burn\_in\_period`.
        This is done to overcome problems with losses quickly changing in the first iterations.

    Attributes
    ----------
    num_tasks
        Number of tasks/heads of the module.
    backbone
        Shared module.
    heads
        Sequence with the (rotated) task-specific heads.
    rep
        Current output of the backbone (after calling forward during training).


    References
    ----------
    .. [1] Javaloy, Adrián, and Isabel Valera. "RotoGrad: Gradient Homogenization in Multitask Learning."
        International Conference on Learning Representations (2022).

    """
    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: Optional[torch.Tensor]

    def __init__(self, backbone: nn.Module, heads: Sequence[nn.Module], rotation_shape: Union[int, torch.Size], *args,
                 post_shape: torch.Size = (), normalize_losses: bool = False, burn_in_period: int = 20):
        super(RotateOnly, self).__init__()
        num_tasks = len(heads)
        if isinstance(rotation_shape, int):
            rotation_shape = torch.Size((rotation_shape,))
        assert len(rotation_shape) > 0
        rotation_size = reduce(int.__mul__, rotation_shape)

        for i in range(num_tasks):
            heads[i] = nn.Sequential(RotateModule(self, i), heads[i])

        self._backbone = [backbone]
        self.heads = heads

        # Parameterize rotations so we can run unconstrained optimization
        for i in range(num_tasks):
            self.register_parameter(f'rotation_{i}', nn.Parameter(torch.eye(rotation_size), requires_grad=True))
            orthogonal(self, f'rotation_{i}', triv='expm')  # uses exponential map (alternative: cayley)

        # Parameters
        self.num_tasks = num_tasks
        self.rotation_shape = rotation_shape
        self.rotation_size = rotation_size
        self.post_shape = post_shape
        self.burn_in_period = burn_in_period
        self.normalize_losses = normalize_losses

        self.rep = None
        self.grads = [None for _ in range(num_tasks)]
        self.original_grads = [None for _ in range(num_tasks)]
        self.losses = [None for _ in range(num_tasks)]
        self.initial_losses = [None for _ in range(num_tasks)]
        self.initial_backbone_loss = None
        self.iteration_counter = 0

    @property
    def rotation(self) -> Sequence[torch.Tensor]:
        r"""List of rotations matrices, one per task. These are trainable, make sure to call `detach()`."""
        return [getattr(self, f'rotation_{i}') for i in range(self.num_tasks)]

    @property
    def backbone(self) -> nn.Module:
        return self._backbone[0]

    def to(self, *args, **kwargs):
        self.backbone.to(*args, **kwargs)
        for head in self.heads:
            head.to(*args, **kwargs)
        return super(RotateOnly, self).to(*args, **kwargs)

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        self.backbone.train(mode)
        for head in self.heads:
            head.train(mode)
        return self

    def __len__(self) -> int:
        r"""Returns the number of tasks."""
        return self.num_tasks

    def __getitem__(self, item) -> nn.Module:
        r"""Returns an end-to-end model for the selected task."""
        return nn.Sequential(self.backbone, self.heads[item])

    def _hook(self, index):
        def _hook_(g):
            self.original_grads[index] = g

        return _hook_

    def forward(self, x: Any) -> Sequence[Any]:
        """Forwards the input `x` through the backbone and all heads, returning a list with all the task predictions.
        It can be thought as something similar to:

        .. code-block:: python

            preds = []
            z = backbone(x)
            for R_i, head in zip(rotations, heads):
                z_i = rotate(R_i, z)
                preds.append(head(z_i))
            return preds

        """
        preds = []
        out = self.backbone(x)

        if isinstance(out, (list, tuple)):
            rep, extra_out = out[0], out[1:]
            extra_out = list(extra_out)
        else:
            rep = out
            extra_out = []

        if self.training:
            self.rep = rep

        for i, head in enumerate(self.heads):
            rep_i = rep
            if self.training:
                rep_i = rep.detach().clone()
                rep_i.requires_grad = True
                rep_i.register_hook(self._hook(i))

            out_i = head(rep_i)
            if isinstance(out_i, (list, tuple)):
                preds.append(out_i[0])
                extra_out.append(out_i[1:])
            else:
                preds.append(out_i)

        if len(extra_out) == 0:
            return preds
        else:
            return preds, extra_out

    def backward(self, losses: Sequence[torch.Tensor], backbone_loss=None, **kwargs) -> None:
        r"""Computes the backward computations for the entire model (that is, shared and specific modules).
        It also computes the gradients for the rotation matrices.

        Parameters
        ----------
        losses
            Sequence of the task losses from which back-propagate.
        backbone_loss
            Loss exclusive for the backbone (for example, a regularization term).
        """
        assert self.training, 'Backward should only be called when training'

        if self.iteration_counter == 0 or self.iteration_counter == self.burn_in_period:
            for i, loss in enumerate(losses):
                self.initial_losses[i] = loss.item()

            if self.normalize_losses and backbone_loss is not None:
                self.initial_backbone_loss = backbone_loss.item()

        self.iteration_counter += 1

        for i in range(len(losses)):
            loss = losses[i] / self.initial_losses[i]
            self.losses[i] = loss.item()

            if self.normalize_losses:
                loss.backward(**kwargs)
            else:
                losses[i].backward(**kwargs)

        if backbone_loss is not None:
            if self.normalize_losses:
                (backbone_loss / self.initial_backbone_loss).backward(retain_graph=True)
            else:
                backbone_loss.backward(retain_graph=True)

        self.rep.backward(self._rep_grad())

    def _rep_grad(self):
        old_grads = self.original_grads  # these grads are already rotated, we have to recover the originals
        grads = self.grads

        # Compute the reference vector
        mean_grad = sum([g for g in old_grads]).detach().clone() / len(grads)
        mean_norm = mean_grad.norm(p=2)
        old_grads2 = [g * divide(mean_norm, g.norm(p=2)) for g in old_grads]
        mean_grad = sum([g for g in old_grads2]).detach().clone() / len(grads)

        dim_post = -len(self.post_shape)
        dim_rot = -len(self.rotation_shape)
        og_shape = mean_grad.shape
        if dim_post == 0:
            mean_grad = mean_grad.unsqueeze(dim=-1)
            dim_post = -1

        mean_grad = mean_grad.flatten(start_dim=dim_post)
        mean_grad = mean_grad.flatten(start_dim=dim_rot - 1, end_dim=-2)

        for i, grad in enumerate(grads):
            R = self.rotation[i]
            loss_rotograd = rotate(mean_grad, R, self.rotation_size).view(og_shape) - grad
            loss_rotograd = loss_rotograd.flatten(start_dim=dim_post)
            loss_rotograd = loss_rotograd.flatten(start_dim=dim_rot - 1, end_dim=-2)
            loss_rotograd = torch.einsum('...ij,...ij->...', loss_rotograd, loss_rotograd)
            loss_rotograd.mean().backward()

        return sum(old_grads)

    def mtl_parameters(self, recurse=True):
        return self.parameters(recurse=recurse)

    def model_parameters(self, recurse=True):
        for param in self.backbone.parameters(recurse=recurse):
            yield param

        for h in self.heads:
            for param in h.parameters(recurse=recurse):
                yield param


class RotoGrad(RotateOnly):
    r"""
    Implementation of RotoGrad as described in the original paper. [1]_

    Parameters
    ----------
    backbone
        Shared module.
    heads
        Task-specific modules.
    rotation_shape
        Shape of the shared representation to be rotated which, usually, is just the size of the backbone's output.
        Passing a shape is useful, for example, if you want to rotate an image with shape width x height.
    post_shape : optional, default=()
        Shape of the shared representation following the part to be rotated (if any). This part will be kept as it is.
        This is useful, for example, if you want to rotate only the channels of an image.
    burn_in_period : optional, default=20
        When back-propagating towards the shared parameters, *each task loss is normalized dividing by its initial
        value*, :math:`{L_k(t)}/{L_k(t_0 = 0)}`. This parameter sets a number of iterations after which the denominator
        will be replaced by the value of the loss at that iteration, that is, :math:`t_0 = burn\_in\_period`.
        This is done to overcome problems with losses quickly changing in the first iterations.
    normalize_losses : optional, default=False
        Whether to use this normalized losses to back-propagate through the task-specific parameters as well.

    Attributes
    ----------
    num_tasks
        Number of tasks/heads of the module.
    backbone
        Shared module.
    heads
        Sequence with the (rotated) task-specific heads.
    rep
        Current output of the backbone (aft1er calling forward during training).


    References
    ----------
    .. [1] Javaloy, Adrián, and Isabel Valera. "RotoGrad: Gradient Homogenization in Multitask Learning."
        International Conference on Learning Representations (2022).

    """
    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: torch.Tensor

    def __init__(self, backbone: nn.Module, heads: Sequence[nn.Module],  rotation_shape: Union[int, torch.Size], *args,
                 post_shape: torch.Size = (), normalize_losses: bool = False, burn_in_period: int = 20):
        super().__init__(backbone, heads, rotation_shape, *args,
                         post_shape=post_shape, burn_in_period=burn_in_period, normalize_losses=normalize_losses)

        self.initial_grads = None
        self.counter = 0

    def _rep_grad(self):
        super()._rep_grad()

        grad_norms = [torch.norm(g, keepdim=True).clamp_min(1e-15) for g in self.original_grads]

        if self.initial_grads is None or self.counter == self.burn_in_period:
            self.initial_grads = grad_norms
        self.counter += 1

        conv_ratios = [x / y for x, y, in zip(grad_norms, self.initial_grads)]
        alphas = [x / torch.clamp(sum(conv_ratios), 1e-15) for x in conv_ratios]

        weighted_sum_norms = sum([a * g for a, g in zip(alphas, grad_norms)])
        grads = [g / n * weighted_sum_norms for g, n in zip(self.original_grads, grad_norms)]
        return sum(grads)


class RotoGradNorm(RotoGrad):
    r"""Implementation of RotoGrad as described in the original paper, [1]_ combined with GradNorm [2]_ to homogeneize
    both the direction and magnitude of the task gradients.

    Parameters
    ----------
    backbone
        Shared module.
    heads
        Task-specific modules.
    rotation_shape
        Shape of the shared representation to be rotated which, usually, is just the size of the backbone's output.
        Passing a shape is useful, for example, if you want to rotate an image with shape width x height.
    alpha
        :math:`\alpha` hyper-parameter as described in GradNorm, [2]_ used to compute the reference direction.
    post_shape : optional, default=()
        Shape of the shared representation following the part to be rotated (if any). This part will be kept as it is.
        This is useful, for example, if you want to rotate only the channels of an image.
    burn_in_period : optional, default=20
        When back-propagating towards the shared parameters, *each task loss is normalized dividing by its initial
        value*, :math:`{L_k(t)}/{L_k(t_0 = 0)}`. This parameter sets a number of iterations after which the denominator
        will be replaced by the value of the loss at that iteration, that is, :math:`t_0 = burn\_in\_period`.
        This is done to overcome problems with losses quickly changing in the first iterations.
    normalize_losses : optional, default=False
        Whether to use this normalized losses to back-propagate through the task-specific parameters as well.
    TODO

    Attributes
    ----------
    num_tasks
        Number of tasks/heads of the module.
    backbone
        Shared module.
    heads
        Sequence with the (rotated) task-specific heads.
    rep
        Current output of the backbone (after calling forward during training).


    References
    ----------
    .. [1] Javaloy, Adrián, and Isabel Valera. "RotoGrad: Gradient Homogenization in Multitask Learning."
        International Conference on Learning Representations (2022).

    .. [2] Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
        International Conference on Machine Learning. PMLR, 2018.

    """

    def __init__(self, backbone: nn.Module, heads: Sequence[nn.Module],  rotation_shape: Union[int, torch.Size], *args,
                 alpha: float, post_shape: torch.Size = (), normalize_losses: bool = False, burn_in_period: int = 20):
        super().__init__(backbone, heads, rotation_shape, *args,
                         post_shape=post_shape, burn_in_period=burn_in_period, normalize_losses=normalize_losses)
        self.alpha = alpha
        self.weight_ = nn.ParameterList([nn.Parameter(torch.ones([]), requires_grad=True) for _ in range(len(heads))])

    @property
    def weight(self) -> Sequence[torch.Tensor]:
        r"""List of task weights, one per task. These are trainable, make sure to call `detach()`."""
        ws = [w.exp() + 1e-15 for w in self.weight_]
        norm_coef = self.num_tasks / sum(ws)
        return [w * norm_coef for w in ws]

    def _rep_grad(self):
        super()._rep_grad()

        grads_norm = [g.norm(p=2) for g in self.original_grads]

        mean_grad = sum([g * w for g, w in zip(self.original_grads, self.weight)]).detach().clone() / len(self.grads)
        mean_grad_norm = mean_grad.norm(p=2)
        mean_loss = sum(self.losses) / len(self.losses)

        for i, [loss, grad] in enumerate(zip(self.losses, grads_norm)):
            inverse_ratio_i = (loss / mean_loss) ** self.alpha
            mean_grad_i = mean_grad_norm * float(inverse_ratio_i)

            loss_gradnorm = torch.abs(grad * self.weight[i] - mean_grad_i)
            loss_gradnorm.backward()

        with torch.no_grad():
            new_grads = [g * w for g, w in zip(self.original_grads, self.weight)]

        return sum(new_grads)
