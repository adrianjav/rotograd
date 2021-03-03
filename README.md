# RotoGrad

---


[comment]: <> ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.2002.11369-B31B1B.svg&#41;]&#40;https://arxiv.org/abs/2002.11369&#41;)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/adrianjav/rotograd/blob/main/LICENSE)

> A library for dynamic gradient homogenization for multitask learning in Pytorch

## Installation

---


Installing this library is as simple as running in your terminal
```bash
pip install rotograd
```

The code has been tested in Pytorch 1.7.0, yet it should work on most versions. Feel free to open an issue
if that were not the case.

## Overview

---


This is the official Pytorch implementation of RotoGrad, an algorithm to reduce the negative transfer due 
to gradient conflict with respect to the shared parameters when different tasks of a multi-task learning
system fight for the shared resources.

Let's say you have a hard-parameter sharing architecture with a `backbone` model shared across tasks, and 
two different tasks you want to solve. These tasks take the output of the backbone `z = backbone(x)` and fed
it to a task-specific model (`head1` and `head2`) to obtain the predictions of their tasks, that is,
`y1 = head1(z)` and `y2 = head2(z)`.

Then you can simply use RotoGrad or RotoGradNorm (RotoGrad + GradNorm) by putting all parts together in a
single model.

```python
from rotograd import RotoGradNorm
model = RotoGradNorm(backbone, [head1, head2], size_z, alpha=1.)
```

where you can recover the backbone and i-th head simply calling `model.backbone` and `model.heads[i]`. Even
more, you can obtain the end-to-end model for a single task (that is, backbone + head), by typing `model[i]`.

As discussed in the paper, it is advisable to have a smaller learning rate for the parameters of RotoGrad
and GradNorm. This is as simple as doing:

```python
optim_model = nn.Adam({'params': m.parameters() for m in [backbone, head1, head2]}, lr=learning_rate_model)
optim_rotograd = nn.Adam({'params': model.parameters()}, lr=learning_rate_rotograd)
```

Finally, we can train the model on all tasks using a simple step function:
```python
import rotograd

def step(x, y1, y2):
    model.train()
    
    optim_model.zero_grad()
    optim_rotograd.zero_grad()

    with rotograd.cached():  # Speeds-up computations by caching Rotograd's parameters
        pred1, pred_2 = model(x)
        
        loss1 = loss_task1(pred1, y1)
        loss2 = loss_task2(pred2, y2)
        
        model.backward([loss1, loss2])
    
    optim_model.step()
    optim_rotograd.step()
        
    return loss1, loss2
```

## Cooperative mode

---

In the main paper, a cooperative version of RotoGrad (and RotoGradNorm) is introduced. 
The intuition is that, after a few epochs where RotoGrad has properly aligned the gradients, it can start
focusing on helping to reduce the tasks loss functions as well. 

Enabling this mode is as simple as calling `model.coop(True/False)` after `T` training epochs. This method works 
similarly  to `.train()` and `.eval()` in Pytorch's Modules, setting a boolean variable to tell RotoGrad
to enable/disable the cooperative mode.

[comment]: <> (## Cite)

[comment]: <> (Consider citing the following paper if you use RotoGrad:)

[comment]: <> (```)

[comment]: <> (```)