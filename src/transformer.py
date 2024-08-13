import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from multihead_attention import MultiHeadAttentionLayer
from feedforward_network import FeedForwardNeuralNetwork
from layer_normalization import LayerNormalization


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.eps = eps
        self.bias = bias

        self.multihead = MultiHeadAttentionLayer(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
            bias=self.bias,
        )

        self.layernorm = LayerNormalization(
            normalized_shape=self.dimension,
            eps=self.eps,
            bias=self.bias,
        )

        self.feedforward = FeedForwardNeuralNetwork(
            in_features=self.dimension,
            out_features=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.multihead(x=x, mask=mask)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)

            x = self.layernorm(x=x)

            residual = x

            x = self.feedforward(x=x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)

            x = self.layernorm(x=x)

            return x

        else:
            raise TypeError("Input must be a Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer Encoder Block for GPT".title()
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=config()["GPT"]["dimension"],
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=config()["GPT"]["nheads"],
        help="Number of heads in the multihead attention".capitalize(),
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=config()["GPT"]["dim_feedforward"],
        help="Dimension of the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["GPT"]["dropout"],
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["GPT"]["activation"],
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=config()["GPT"]["bias"],
        help="Whether to use bias in the linear layers".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["embedding"]["batch_size"]
    block_size = config()["embedding"]["block_size"]

    transformer = TransformerEncoderBlock(
        dimension=args.dimension,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        bias=args.bias,
    )

    assert transformer(torch.randn(batch_size, block_size, args.dimension)).size() == (
        batch_size,
        block_size,
        args.dimension,
    ), "TransformerEncoderBlock is not working correctly".capitalize()
