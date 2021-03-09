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

   rotograd/rotograd.rst
   rotograd/rotogradnorm.rst
   rotograd/cached.rst



Welcome to |project|'s documentation!
=====================================

|project| is a solution to alleviate the problem of gradient conflict (a.k.a.
gradient interference) in multi-task systems, produced by the gradients of different tasks
w.r.t. the shared parameters pointing towards different directions.

|project| homogeneizes the direction of these gradients during training by accordingly rotating
the input space of the task-specific modules (a.k.a. heads) such that their gradients for the 
shared module (a.k.a. backbone) don't cancel each other out.

This is a Pytorch implementation. For more information you can check out `the pre-print`_.


Installation
------------

Install |project| by running:

.. code-block:: bash

    pip install rotograd
    
How to use
----------

Suppose you have a :python:`backbone` model shared across tasks, and two different tasks to solve. These tasks take the output of the backbone :python:`z = backbone(x)` and fed it to a task-specific model (:python:`head1` and :python:`head2`) to obtain the predictions of their tasks, that is,
:python:`y1 = head1(z)` and :python:`y2 = head2(z)`.

Then you can simply use RotoGrad or RotoGradNorm (RotoGrad + GradNorm) by putting all parts together 
in a single model.

.. code-block:: python

    from rotograd import RotoGradNorm
    model = RotoGradNorm(backbone, [head1, head2], size_z, alpha=1.)

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

        with rotograd.cached():  # Speeds-up computations
            pred1, pred_2 = model(x)
            
            loss1 = loss_task1(pred1, y1)
            loss2 = loss_task2(pred2, y2)
            
            model.backward([loss1, loss2])
        
        optim_model.step()
        optim_rotograd.step()
            
        return loss1, loss2

Cooperative mode
^^^^^^^^^^^^^^^^

|project| has a cooperative mode. The intuition is that, after a few epochs where RotoGrad has properly aligned the gradients, it can start focusing on helping to reduce the tasks loss functions as well. 

Enabling this mode is as simple as calling :python:`model.coop(True/False)` after :python:`T` training epochs. This method works 
similarly  to :python:`model.train()` and :python:`model.eval()` in Pytorch's Modules, setting a boolean variable to tell |project| to enable/disable the cooperative mode.

    
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

    @article{javaloy2021rotograd,
        title={Rotograd: Dynamic Gradient Homogenization for Multi-Task Learning},
        author={Javaloy, Adri\'an and Valera, Isabel},
        journal={arXiv preprint arXiv:2103.02631},
        year={2021}
    }

License
-------

The project is licensed under the MIT license.

.. References and variables

.. |project| replace:: RotoGrad
.. _the pre-print: https://arxiv.org/abs/2103.02631


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
