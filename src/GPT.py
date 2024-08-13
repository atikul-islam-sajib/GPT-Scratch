import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from transformer import TransformerEncoderBlock


class GPT(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super(GPT, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.eps = eps
        self.bias = bias

        self.model = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    dimension=self.dimension,
                    nheads=self.nheads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                    eps=self.eps,
                    bias=self.bias,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            for layer in self.model:
                x = layer(x=x, mask=mask)

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    GPTModel = GPT()

    print(GPTModel(torch.randn(40, 200, 512)).size())
