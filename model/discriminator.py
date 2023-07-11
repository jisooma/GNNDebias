#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 17:23
# @Author  : zhixiuma
# @File    : discriminator.py
# @Project : Test
# @Software: PyCharm
import torch
from torch.nn import Linear

class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h):
        h = self.lin(h)

        return torch.sigmoid(h)

# class MLP_discriminator_1(torch.nn.Module):
#     def __init__(self, args):
#         super(MLP_discriminator_1, self).__init__()
#         self.args = args
#
#         self.lin = Linear(args.hidden, 1)
#
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#
#     def forward(self, h):
#         h = self.lin(h)
#
#         return torch.sigmoid(h)