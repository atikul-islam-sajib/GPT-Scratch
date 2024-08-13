import sys
import torch
import argparse
from tqdm import tqdm
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
                for _ in tqdm(range(self.num_layers))
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
    parser = argparse.ArgumentParser(description="GPT model".title())
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
        "--num_layers",
        type=int,
        default=config()["GPT"]["num_layers"],
        help="Number of layers in the transformer".capitalize(),
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

    GPTModel = GPT(
        dimension=args.dimension,
        nheads=args.nheads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        bias=args.bias,
    )

    assert GPTModel(torch.randn(batch_size, block_size, args.dimension)).size() == (
        batch_size,
        block_size,
        args.dimension,
    ), "GPT Model is not working correctly".upper()
