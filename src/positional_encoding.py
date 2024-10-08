import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length: int = 200, dimension: int = 512):
        super(PositionalEncoding, self).__init__()

        self.sequence_length = sequence_length
        self.dimension = dimension

        self.positional_encoding = torch.zeros((self.sequence_length, self.dimension))

        for position in range(self.sequence_length):
            for index in range(self.dimension):
                if index % 2 == 0:
                    self.positional_encoding[position, index] = math.sin(
                        position / (10000 ** ((2 * index) / dimension))
                    )
                else:
                    self.positional_encoding[position, index] = math.cos(
                        position / (10000 ** ((2 * index) / dimension))
                    )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.positional_encoding.unsqueeze(0)[:, : x.size(1), :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Positional Encoding for GPT".title())
    parser.add_argument(
        "--block_size",
        type=int,
        default=config()["embedding"]["block_size"],
        help="Block size or sequence length".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=config()["GPT"]["dimension"],
        help="Dimension of the embedding".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["embedding"]["batch_size"]

    positional = PositionalEncoding(
        sequence_length=args.block_size, dimension=args.dimension
    )

    assert positional(
        torch.randn(batch_size, args.block_size, args.dimension)
    ).size() == (
        batch_size // batch_size,
        args.block_size,
        args.dimension,
    ), "Positional encoding is not working properly".capitalize()
