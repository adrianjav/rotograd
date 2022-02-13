.. rotograd documentation master file, created by
   sphinx-quickstart on Tue Mar  9 09:52:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. role:: python(code)
    :language: python

.. toctree::
   :caption: RotoGrad
   :hidden:

   Introduction <self>
   
.. toctree::
   :maxdepth: 2
   :caption: RotoGrad API
   :hidden:

   rotograd/rotateonly.rst
   rotograd/rotograd.rst
   rotograd/rotogradnorm.rst
   rotograd/cached.rst



Welcome to |project|'s documentation!
=====================================

|project| is a solution to alleviate the problem of gradient conflict (a.k.a.
gradient interference) in multitask models, produced by the gradients of different tasks
w.r.t. the shared parameters pointing towards different directions.

|project| homogeneizes the gradients of these gradients during training by accordingly scaling and rotating
the input space of the task-specific modules (a.k.a. heads) such that their gradients for the 
shared module (a.k.a. backbone) do not overweight/cancel each other out.

This is a Pytorch implementation. For more information you can check out `the original paper`_.


Installation
------------

Install |project| by running:

.. code-block:: bash

    pip install rotograd
    
How to use
----------

Suppose you have a :python:`backbone` model shared across tasks, and two different tasks to solve. These tasks take the
output of the backbone :python:`z = backbone(x)` and fed it to a task-specific model (:python:`head1`
and :python:`head2`) to obtain the predictions of their tasks, that is,
:python:`y1 = head1(z)` and :python:`y2 = head2(z)`.

Then you can simply use RotateOnly, RotoGrad. or RotoGradNorm (RotateOnly + GradNorm) by putting all parts together in
a single model.

.. code-block:: python

    from rotograd import RotoGrad
   model = RotoGrad(backbone, [head1, head2], size_z, normalize_losses=True)

where you can recover the backbone and i-th head simply calling :python:`model.backbone` and
:python:`model.heads[i]`. Even more, you can obtain the end-to-end model for a single task (that is,
backbone + head), by typing :python:`model[i]`.

As discussed in the paper, it is advisable to have a smaller learning rate for the parameters of 
RotoGrad and GradNorm. This is as simple as doing:

.. code-block:: python

   optim_model = nn.Adam({'params': m.parameters() for m in [backbone, head1, head2]}, lr=learning_rate_model)
   optim_rotograd = nn.Adam({'params': model.parameters()}, lr=learning_rate_rotograd)


Finally, we can train the model on all tasks using a simple step function:

.. code-block:: python

   import rotograd

   def step(x, y1, y2):
       model.train()

       optim_model.zero_grad()
       optim_rotograd.zero_grad()

       with rotograd.cached():  # Speeds-up computations by caching Rotograd's parameters
           pred1, pred2 = model(x)

           loss1 = loss_task1(pred1, y1)
           loss2 = loss_task2(pred2, y2)

           model.backward([loss1, loss2])

       optim_model.step()
       optim_rotograd.step()

       return loss1, loss2

    
Contribute
----------

- Issue Tracker: https://github.com/adrianjav/rotograd/issues
- Source Code: https://github.com/adrianjav/rotograd

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: adrian.javaloy@gmail.com

Citing
-------
.. code-block:: bibtex

   @inproceedings{javaloy2022rotograd,
      title={RotoGrad: Gradient Homogenization in Multitask Learning},
      author={Adri{\'a}n Javaloy and Isabel Valera},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=T8wHz4rnuGL}
   }

License
-------

The project is licensed under the MIT license.

.. References and variables

.. |project| replace:: RotoGrad
.. _the original paper: https://arxiv.org/abs/2103.02631


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
