import sys
import math
import torch
import argparse
import torch.nn as nn


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

        print("done")

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.positional_encoding.unsqueeze(0)[:, : x.size(1), :]


if __name__ == "__main__":
    positional = PositionalEncoding(sequence_length=200, dimension=512)
    print(positional(torch.randn(40, 200, 512)).size())
