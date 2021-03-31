from typing import Sequence, Any

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

    def _hook(self, index):
        def _hook_(g):
            self.grads[index] = g
        return _hook_

    def forward(self, x):
        preds = []
        self.rep = self.backbone(x)

        for i, head in enumerate(self.heads):
            rep_i = self.rep.detach().clone()
            rep_i.requires_grad = True
            rep_i.register_hook(self._hook(i))

            preds.append(head(rep_i))

        return preds

    def backward(self, losses):
        for loss in losses:
            loss.backward()
        self.rep.backward(sum(self.grads))


def rotate(points, rotation):
    return torch.einsum('ij,bj->bi', rotation, points)


def rotate_back(points, rotation):
    return rotate(points, rotation.t())


class RotateModule(nn.Module):
    def __init__(self, parent, item):
        super().__init__()

        self.parent = [parent]  # Dirty trick to don't register parameters TODO improve
        self.item = item

    def hook(self, g):
        self.p.grads[self.item] = g.clone()

    @property
    def p(self):
        return self.parent[0]

    @property
    def R(self):
        return self.p.rotation[self.item]

    @property
    def weight(self):
        return self.p.weight[self.item] if hasattr(self.p, 'weight') else 1.

    def rotate(self, z):
        return rotate(z, self.R)

    def rotate_back(self, z):
        return rotate_back(z, self.R)

    def forward(self, z):
        def r_hook(g):
            self.p.R_hook1[self.item] = g

        R = self.R.clone().detach()
        if self.p._coop:
            R.requires_grad = True
            if self.p.training:
                R.register_hook(r_hook)

        new_z = rotate(z, R)
        if self.p.training:
            new_z.register_hook(self.hook)

        return new_z


class RotoGrad(nn.Module):
    r"""
    Implementation of RotoGrad as described in the original paper. [1]_

    Parameters
    ----------
    backbone
        Shared module.
    heads
        Task-specific modules.
    latent_size
        Size of the shared representation, that is, size of the output of backbone.
    alpha
        :math:`\alpha` hyper-parameter as described in GradNorm, [2]_ used to compute the reference direction.
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
        Current output of the backbone (after calling forward during training).


    References
    ----------
    .. [1] Javaloy, Adrián, and Isabel Valera. "Rotograd: Dynamic Gradient Homogenization for Multi-Task Learning."
        arXiv preprint arXiv:2103.02631 (2021).

    .. [2] Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
        International Conference on Machine Learning. PMLR, 2018.

    """
    num_tasks: int
    backbone: nn.Module
    heads: Sequence[nn.Module]
    rep: torch.Tensor

    def __init__(self, backbone: nn.Module, heads: Sequence[nn.Module], latent_size: int, *args, alpha: float,
                 burn_in_period: int = 20, normalize_losses: bool = False):
        super(RotoGrad, self).__init__()
        num_tasks = len(heads)

        for i in range(num_tasks):
            heads[i] = nn.Sequential(RotateModule(self, i), heads[i])

        self._backbone = [backbone]
        self.heads = heads

        # Parameterize rotations so we can run unconstrained optimization
        for i in range(num_tasks):
            self.register_parameter(f'rotation_{i}', nn.Parameter(torch.eye(latent_size), requires_grad=True))
            orthogonal(self, f'rotation_{i}', triv='expm')  # uses exponential map (alternative: cayley)

        # Parameters
        self.num_tasks = num_tasks
        self.latent_size = latent_size
        self.alpha = alpha
        self._coop = False
        self.burn_in_period = burn_in_period
        self.normalize_losses = normalize_losses

        self.rep = None
        self.grads = [None for _ in range(num_tasks)]
        self.original_grads = [None for _ in range(num_tasks)]
        self.R_hook1 = [None for _ in range(num_tasks)]
        self.R_hook2 = [None for _ in range(num_tasks)]
        self.losses = [None for _ in range(num_tasks)]
        self.initial_losses = [None for _ in range(num_tasks)]
        self.iteration_counter = 0

    @property
    def rotation(self) -> Sequence[torch.Tensor]:
        r"""List of rotations matrices, one per task. These are trainable, make sure to call `detach()`."""
        return [getattr(self, f'rotation_{i}') for i in range(self.num_tasks)]

    @property
    def backbone(self) -> nn.Module:
        return self._backbone[0]

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        self.backbone.train(mode)
        for head in self.heads:
            head.train(mode)
        return self

    def coop(self, value: bool = True) -> nn.Module:
        r"""Enables/disables the RotoGrad's cooperative mode."""
        self._coop = value
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
        self.rep = self.backbone(x)

        for i, head in enumerate(self.heads):
            rep_i = self.rep.detach().clone()
            rep_i.requires_grad = True
            rep_i.register_hook(self._hook(i))

            preds.append(head(rep_i))

        return preds

    def backward(self, losses: Sequence[torch.Tensor]) -> None:
        r"""Computes the backward computations for the entire model (that is, shared and specific modules).
        It also computes the gradients for the rotation matrices.

        Parameters
        ----------
        losses
            Sequence of the task losses from which back-propagate.
        """
        if self.training:
            if self.iteration_counter == 0 or self.iteration_counter == self.burn_in_period:
                for i, loss in enumerate(losses):
                    self.initial_losses[i] = loss.item()

            self.iteration_counter += 1

        for i in range(len(losses)):
            loss = losses[i] / self.initial_losses[i]
            self.losses[i] = loss.item()

            if self.normalize_losses:
                loss.backward()
            else:
                losses[i].backward()

        self.rep.backward(self._rep_grad)

    @property
    def _rep_grad(self):
        old_grads = self.original_grads  # these grads are already rotated, we have to recover the originals
        # with torch.no_grad():
        #     grads = [rotate(g, R) for g, R in zip(grads, self.rotation)]
        #
        grads = self.grads

        mean_grad = sum([g for g in old_grads]).detach().clone() / len(grads)
        mean_loss = sum(self.losses) / len(self.losses)

        # Compute the reference vector
        mean_norm = mean_grad.norm(p=2)
        inverse_ratios = [(loss / mean_loss) ** self.alpha for loss in self.losses]
        old_grads2 = [g * divide(mean_norm * ir, g.norm(p=2)) for g, ir in zip(old_grads, inverse_ratios)]
        mean_grad = sum([g for g in old_grads2]).detach().clone() / len(grads)

        for i, grad in enumerate(grads):
            def r_hook(g):
                self.R_hook2[i] = g

            R = self.rotation[i]
            if self._coop:
                R = R.clone().detach()
                R.requires_grad = True
                R.register_hook(r_hook)

            loss_rotograd = rotate(mean_grad, R) - grad
            loss_rotograd = torch.einsum('bi,bi->b', loss_rotograd, loss_rotograd)
            loss_rotograd.mean().backward()

            if self._coop:
                mean_grad_norm = (self.R_hook1[i] + self.R_hook2[i]) * 0.5
                mean_grad_norm = torch.norm(mean_grad_norm, p=2)
                weight1 = divide(mean_grad_norm, torch.norm(self.R_hook1[i], p=2))
                weight2 = divide(mean_grad_norm, torch.norm(self.R_hook2[i], p=2))
                self.rotation[i].backward(self.R_hook1[i] * weight1 + self.R_hook2[i] * weight2)

        return sum(old_grads)


class RotoGradNorm(RotoGrad):
    r"""Implementation of RotoGrad as described in the original paper, [1]_ combined with GradNorm [2]_ to homogeneize
    both the direction and magnitude of the task gradients.

    Parameters
    ----------
    backbone
        Shared module.
    heads
        Task-specific modules.
    latent_size
        Size of the shared representation, that is, size of the output of backbone.
    alpha
        :math:`\alpha` hyper-parameter as described in GradNorm, [2]_ used to compute the reference direction.
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
        Current output of the backbone (after calling forward during training).


    References
    ----------
    .. [1] Javaloy, Adrián, and Isabel Valera. "Rotograd: Dynamic Gradient Homogenization for Multi-Task Learning."
        arXiv preprint arXiv:2103.02631 (2021).

    .. [2] Chen, Zhao, et al. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks."
        International Conference on Machine Learning. PMLR, 2018.

    """

    def __init__(self, backbone: nn.Module, heads: Sequence[nn.Module], latent_size: int, *args, alpha: float,
                 burn_in_period: int = 20, normalize_losses: bool = False):
        super().__init__(backbone, heads, latent_size, *args, alpha=alpha, burn_in_period=burn_in_period,
                         normalize_losses=normalize_losses)
        self.weight_ = nn.ParameterList([nn.Parameter(torch.ones([]), requires_grad=True) for _ in range(len(heads))])

    @property
    def weight(self) -> Sequence[torch.Tensor]:
        r"""List of task weights, one per task. These are trainable, make sure to call `detach()`."""
        ws = [w.exp() + 1e-15 for w in self.weight_]
        with torch.no_grad():
            norm_coef = self.num_tasks / sum(ws)
        return [w * norm_coef for w in ws]

    @property
    def _rep_grad(self):
        super()._rep_grad

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
