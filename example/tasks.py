from collections import namedtuple

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score


class Task(namedtuple('Task', ['name', 'loss', 'metric', 'weight', 'index'])):
    __slots__ = ()


def error_rate(pred, target):
    def to_number(x):
        return (x > 0.5).long() if (len(x.size()) == 1 or x.size(1) == 1) else torch.argmax(pred, dim=1)

    return (to_number(pred) == target).float().mean()


def f1(pred, target):
    def to_number(x):
        return (x > 0.5).long() if (len(x.size()) == 1 or x.size(1) == 1) else torch.argmax(pred, dim=1)

    return torch.tensor(f1_score(target.flatten().long().cpu(), to_number(pred.flatten()).cpu()))


def get_function(key):
    def toy(z, t, sign):
        assert sign in ['plus', 'minus']
        sign = 1. if sign == 'plus' else -1.

        return - ((z[..., 0] + sign * 1.5).reciprocal() * torch.sin(3 * (z[..., 0] + sign * 1.5)) + (
               z[..., 1] + sign * 1.5).reciprocal() * torch.sin(3 * (z[..., 1] + sign * 1.5))) + \
               (z + sign * 1.5).norm(p=1, dim=-1)

    functions = {
        'mse': F.mse_loss,
        'bce': F.binary_cross_entropy,
        'bce_logits': F.binary_cross_entropy_with_logits,
        'acc': error_rate,
        'auc': lambda pred, target: torch.tensor(roc_auc_score(target.cpu(), pred.cpu())),
        'nll': lambda i, t: F.nll_loss(i, t.long()),
        'toy1': lambda z, t: toy(z, t, sign='minus'),
        'toy2': lambda z, t: toy(z, t, sign='plus'),
        'f1': f1,
    }

    return functions[key]


def get_tasks(task_names, weights, dataset):
    if task_names == 'all':
        task_names = dataset.tasks.keys()

    if weights == 'uniform':
        weights = [1./len(task_names)] * len(task_names)

    sum_weights = len(task_names) / sum(weights)

    tasks, check = [], 0.
    for name, weight_i in zip(task_names, weights):
        assert name in dataset.tasks.keys(), f'task {name} does not exist for dataset {type(dataset).__name__}.'
        index, losses = dataset.tasks[name]

        if type(losses) == str:
            losses = [losses, losses]
        elif len(losses) == 1:
            losses = [losses[0], losses[0]]

        loss_i, metric_i = losses

        loss_i = get_function(loss_i)
        metric_i = get_function(metric_i)

        loss_i.__name__ = losses[0]
        metric_i.__name__ = losses[1]

        weight_i = weight_i * sum_weights
        task_i = Task(name, loss_i, metric_i, weight_i, index)
        check += weight_i

        tasks.append(task_i)

    assert (check - len(tasks)) < 1e-3
    return tasks


def tasks_parameters(tasks):
    for task_i in tasks:
        yield task_i.weight
