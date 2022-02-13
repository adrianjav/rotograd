from functools import partial
import itertools

import pickle
import torch
import torch.optim

from ignite.engine import Engine, Events
from ignite.metrics import Average, RunningAverage
from ignite.handlers import TerminateOnNan, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

import matplotlib.pyplot as plt
from plot import setup_grid, plot_toy

from rotograd import cached


def create_trainer(model, tasks, optims, loaders, args):
    zt = []
    zt_task = {'left': [], 'right': []}

    if args.dataset.name == 'dummy':
        lim = 2.5
        lims = [[-lim, lim], [-lim, lim]]
        grid = setup_grid(lims, 1000)

    def trainer_step(engine, batch):
        model.train()

        for optim in optims:
            optim.zero_grad()

        # Batch data
        x, y = batch
        x = convert_tensor(x.float(), args.device)
        y = [convert_tensor(y_, args.device) for y_ in y]

        training_loss = 0.
        losses = []

        # Intermediate representation
        with cached():
            preds = model(x)
            if args.dataset.name == 'dummy':
                zt.append(model.rep.detach().clone())

            for pred_i, task_i in zip(preds, tasks):
                loss_i = task_i.loss(pred_i, y[task_i.index])

                if args.dataset.name == 'dummy':
                    loss_i = loss_i.mean(dim=0)
                    zt_task[task_i.name].append(pred_i.detach().clone())

                # Track losses
                losses.append(loss_i)
                training_loss += loss_i.item() * task_i.weight

            if args.dataset.name == 'dummy' and (engine.state.epoch == engine.state.max_epochs or engine.state.epoch % args.training.plot_every == 0):
                fig = plot_toy(grid, model, tasks, [zt, zt_task['left'], zt_task['right']],
                               trainer.state.iteration - 1, levels=20, lims=lims)
                fig.savefig(f'plots/step_{engine.state.iteration - 1}.png')
                plt.close(fig)

            model.backward(losses)
            
        for optim in optims:  # Run the optimizers
            optim.step()

        return training_loss, losses

    trainer = Engine(trainer_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    for i, task_i in enumerate(tasks):
        output_transform = partial(lambda idx, x: x[1][idx], i)
        RunningAverage(output_transform=output_transform).attach(trainer, f'train_{task_i.name}')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=['loss'] + [f'train_{t.name}' for t in tasks])

    # Validation
    validator = create_evaluator(model, tasks, args)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validator(trainer):
        validator.run(loaders['val'])
        metrics = validator.state.metrics
        loss = 0.
        for task_i in tasks:
            loss += metrics[f'loss_{task_i.name}'] * task_i.weight

        trainer.state.metrics['val_loss'] = loss

    # Checkpoints
    model_checkpoint = {'model': model}
    handler = ModelCheckpoint('checkpoints', 'latest', require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.training.save_every), handler, model_checkpoint)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.training.save_every))
    def save_state(engine):
        with open('checkpoints/state.pkl', 'wb') as f:
            pickle.dump(engine.state, f)

    @trainer.on(Events.COMPLETED(every=args.training.save_every))
    def save_state(engine):
        with open('checkpoints/state.pkl', 'wb') as f:
            pickle.dump(engine.state, f)

    handler = ModelCheckpoint('checkpoints', 'best', require_empty=False,
                              score_function=(lambda e: -e.state.metrics['val_loss']))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.training.save_every), handler, model_checkpoint)
    trainer.add_event_handler(Events.COMPLETED, handler, model_checkpoint)

    return trainer


def create_evaluator(model, tasks, args):
    model.to(args.device)

    @torch.no_grad()
    def evaluator_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(args.device)
        y = [_y.to(device=args.device) for _y in y]

        losses = {}
        preds = model(x)
        for rep_i, task_i in zip(preds, tasks):
            losses[f'loss_{task_i.name}'] = task_i.loss(rep_i, y[task_i.index]).mean(dim=0)
            losses[f'metric_{task_i.name}'] = task_i.metric(rep_i, y[task_i.index]).mean(dim=0)

        preds = [pred_i.detach().clone() for pred_i in preds]
        return losses, y, preds

    evaluator = Engine(evaluator_step)
    for task_i in tasks:
        for prefix in ['metric', 'loss']:
            name = f'{prefix}_{task_i.name}'
            output_transform = partial(lambda name, x: x[0][name], name)
            Average(output_transform=output_transform).attach(evaluator, name)

    return evaluator


def create_trainer_and_evaluator(model, tasks, optim, loaders, args):
    trainer = create_trainer(model, tasks, optim, loaders, args)
    evaluator = create_evaluator(model, tasks, args)

    return trainer, evaluator


def decorate_trainer(trainer, args, model, schedulers):
    @trainer.on(Events.ITERATION_COMPLETED)
    def apply_schedulers(engine):
        for i, sched in enumerate(schedulers):
            sched.step()


def decorate_evaluator(evaluator, tasks, args):
    @evaluator.on(Events.COMPLETED)
    def log_metrics(evaluator):
        rep_tasks = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in tasks))
        metrics = {f'{k[7:]}_{t.metric.__name__}': v for (k, v), t in zip(evaluator.state.metrics.items(), rep_tasks) if
                   k.startswith('metric_')}

        if hasattr(args, 'single_task_performance') and len(args.single_task_performance) == len(tasks):
            mtl_perf = 0.
            for task_i in tasks:
                single_perf_i, lower_is_better = args.single_task_performance[task_i.name]
                task_perf_i = (evaluator.state.metrics[f'metric_{task_i.name}'] - single_perf_i) / single_perf_i
                if lower_is_better:
                    task_perf_i = -task_perf_i

                mtl_perf += task_perf_i
            mtl_perf /= len(tasks)
            metrics['mtl_perf'] = mtl_perf

        print(f'Metrics: {metrics}')
