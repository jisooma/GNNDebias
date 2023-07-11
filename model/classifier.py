#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 17:19
# @Author  : zhixiuma
# @File    : classifier.py
# @Project : Test
# @Software: PyCharm
import torch
from torch.nn import Linear

class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h