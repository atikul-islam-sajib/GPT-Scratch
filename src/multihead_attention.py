import os
import torch
import argparse
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dropout = dropout
        self.bias = bias

        self.dimension % self.nheads == 0, "dimension must be divisible by nheads".capitalize()

    def forward(self, x: torch.Tensor, mask=None):
        pass
