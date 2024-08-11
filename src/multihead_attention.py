import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from scaled_dot_product import scaled_dot_product_attention


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

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            QKV = self.QKV(x)

            query, key, value = torch.chunk(input=QKV, chunks=3, dim=-1)

            assert (
                query.size() == key.size() == value.size()
            ), "QKV must have the same size".capitalize()

            query = query.view(
                query.size(0), query.size(1), self.nheads, self.dimension // self.nheads
            )
            key = key.view(
                key.size(0), key.size(1), self.nheads, self.dimension // self.nheads
            )
            value = value.view(
                value.size(0), value.size(1), self.nheads, self.dimension // self.nheads
            )

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            attention = scaled_dot_product_attention(
                query=query, key=key, value=value, mask=mask
            )

            assert (
                attention.size() == query.size() == key.size() == value.size()
            ), "Attention must have the same size as QKV".capitalize()

            attention = attention.view(
                attention.size(0),
                attention.size(2),
                attention.size(1) * attention.size(3),
            )

            return attention


if __name__ == "__main__":
    attention = MultiHeadAttentionLayer(dimension=512, nheads=8, dropout=0.1, bias=True)

    print(attention(torch.randn(40, 200, 512)).size())
