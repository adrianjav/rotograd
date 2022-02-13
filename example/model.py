from functools import reduce

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, activation_fn, drop_last=True, bias=True,
                 batch_norm=False, dropout=None):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        if num_layers == 0:
            self.net = activation_fn
            return

        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers):
            layer_i = [nn.Linear(sizes[i], sizes[i+1], bias=bias)]

            if dropout is not None:
                layer_i.append(nn.Dropout(dropout))

            layer_i.append(activation_fn)
            layers.append(layer_i)

        if drop_last:
            layers[-1] = layers[-1][:1]

        if batch_norm:
            layers[-1].append(nn.BatchNorm1d(output_size))

        layers = reduce(list.__add__, layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))

    def last_layer(self):
        for i in range(len(self.net)):
            if isinstance(self.net[-1-i], nn.Linear):
                return self.net[-1-i].parameters()


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

        self.param1 = nn.Parameter(torch.ones([]), requires_grad=True)
        self.param2 = nn.Parameter(torch.ones([]), requires_grad=True)

    def forward(self, input):
        x = input[..., 0] * self.param1
        y = input[..., 1] * self.param2

        return torch.stack((x, y), dim=1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.input_size = 2

    def forward(self, input):
        return input