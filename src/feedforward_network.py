import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation
        self.bias = bias

        if self.activation == "relu":
            self.activation_fn = nn.ReLU(inplace=True)

        elif self.activation == "gelu":
            self.activation_fn = nn.GELU()

        elif self.activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.layers = list()

        for index in range(2):
            self.layers.append(
                nn.Linear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=self.bias,
                )
            )
            if index == 0:
                self.layers.append(self.activation_fn)
                self.layers.append(nn.Dropout(p=self.dropout))

            self.in_features = self.out_features
            self.out_features = in_features

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FeedForwardNeuralNetwork for GPT".capitalize()
    )
    parser.add_argument(
        "--in_features",
        type=int,
        default=config()["GPT"]["dimension"],
        help="Input features",
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=config()["GPT"]["dimension"],
        help="Output features".capitalize(),
    )
    network = FeedForwardNeuralNetwork(
        in_features=512,
        out_features=2048,
        activation="gelu",
        dropout=0.1,
        bias=True,
    )

    print(network(torch.randn(400, 200, 512)).size())
