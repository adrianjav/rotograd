import os
import sys

import yaml
from argparse import Namespace
import torch
import torch.nn as nn
import seaborn as sns

from torch.utils.data import DataLoader

from dataset import DummyDataset
from model import FeedForward
from engine import create_trainer_and_evaluator, decorate_trainer, decorate_evaluator
from tasks import get_tasks

from rotograd import RotoGrad, RotateOnly, VanillaMTL


def get_dataloader(batch_size, **kwargs):
    train_dataset = DummyDataset(tag='train', **kwargs)
    val_dataset = DummyDataset(tag='val', **kwargs)
    test_dataset = DummyDataset(tag='test', **kwargs)

    dataloader_options = {'shuffle': True, 'drop_last': False, 'num_workers': 1}
    if isinstance(batch_size, int):
        batch_size = [batch_size, batch_size]

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size[0], **dataloader_options),
        'val': DataLoader(val_dataset, batch_size=batch_size[1], **dataloader_options),
        'test': DataLoader(test_dataset, batch_size=batch_size[1],  **dataloader_options),
    }

    return loaders


def build_dense(tasks, args):
    activations = {
        'identity': nn.Identity(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }

    heads = []
    shared = getattr(args, 'shared', False)

    enc_params = getattr(args, 'encoder', args)
    backbone = FeedForward(args.input_size, args.rotation_size, enc_params.hidden_size, enc_params.num_layers,
                           activations[enc_params.activation])

    dec_params = getattr(args, 'decoder', args)
    if isinstance(dec_params.hidden_size, int) and not shared:
        dec_params.hidden_size = [dec_params.hidden_size] * len(tasks)
    if isinstance(dec_params.output_size, int) and not shared:
        dec_params.output_size = [dec_params.output_size] * len(tasks)
    if isinstance(dec_params.num_layers, int) and not shared:
        dec_params.num_layers = [dec_params.num_layers] * len(tasks)

    for i, task_i in enumerate(tasks):
        heads.append(FeedForward(args.rotation_size, dec_params.output_size[i], dec_params.hidden_size[i],
                                 dec_params.num_layers[i], activations[dec_params.activation],
                                 dec_params.drop_last or task_i.loss == 'mse'))

    return backbone, heads


def validate_args(args):
    if not hasattr(args, 'seed'):
        args.seed = None

    if not hasattr(args, 'device'):
        args.device = 'cpu'

    if 'cuda' == args.device and not torch.cuda.is_available():
        print('CUDA requested but not available.', file=sys.stderr)
        args.device = 'cpu'

    if not hasattr(args.algorithms, 'learning_rate'):
        args.algorithms.learning_rate = args.training.learning_rate

    if not hasattr(args.algorithms, 'decay'):
        args.algorithms.decay = 1.0


def get_optimizers(model, args):
    all_optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    params_model = [{'params': m.parameters()} for m in [model.backbone] + model.heads]
    params_leader = [{'params': model.parameters()}]

    optimizers = [all_optimizers[args.training.optimizer](params_model, lr=args.training.learning_rate)]
    schedulers = []

    if not isinstance(model, VanillaMTL):
        lr = args.algorithms.learning_rate
        if args.algorithms.learning_rate > 0:
            optimizers.append(all_optimizers[args.algorithms.optimizer](params_leader, lr=lr))
            schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizers[1], args.algorithms.decay))

    return optimizers, schedulers


def main(args):
    validate_args(args)

    loaders = get_dataloader(args.training.batch_size, **vars(args.dataset.options))
    args.model.input_size = loaders['train'].dataset.input_size

    tasks = get_tasks(args.tasks.names, args.tasks.weights, loaders['train'].dataset)
    backbone, heads = build_dense(tasks, args.model)

    if not hasattr(args.rotograd, 'rotation_size'):
        args.rotograd.rotation_size = backbone.output_size

    method = args.algorithms.method
    if method == 'rotograd':
        model = RotoGrad(backbone, heads, args.rotograd.rotation_size, normalize_losses=args.rotograd.normalize)
    elif method == 'rotate':
        model = RotateOnly(backbone, heads, args.rotograd.rotation_size, normalize_losses=args.rotograd.normalize)
    else:
        model = VanillaMTL(backbone, heads)  # TODO add normalize_losses

    print(model)
    model.to(args.device)

    optimizers, schedulers = get_optimizers(model, args)
    trainer, evaluator = create_trainer_and_evaluator(model, tasks, optimizers, loaders, args)

    decorate_trainer(trainer, args, model, schedulers)
    decorate_evaluator(evaluator, tasks, args)

    trainer.run(loaders['train'], max_epochs=args.training.epochs)

    print("Model metrics")
    evaluator.run(loaders['test'])


def to_namespace(args):
    for k, v in args.items():
        if type(v) == dict:
            args[k] = to_namespace(v)
    return Namespace(**args)


if __name__ == '__main__':
    sns.set_style('white')
    with open('toy.yml', 'r') as f:
        args = yaml.safe_load(f)
        args = to_namespace(args)

    os.makedirs(f'results/{args.exp_name}/plots', exist_ok=True)
    os.chdir(f'results/{args.exp_name}')

    main(args)
