import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

sys.path.append("./src/")

from utils import dump, config


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
        default=config()["GPT"]["dim_feedforward"],
        help="Output features".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=config()["GPT"]["activation"],
        help="Activation function".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["GPT"]["dropout"],
        help="Define the dropout rate".capitalize(),
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=config()["GPT"]["bias"],
        help="Define if bias is used".capitalize(),
    )
    args = parser.parse_args()

    batch_size = config()["embedding"]["batch_size"]
    block_size = config()["embedding"]["block_size"]

    network = FeedForwardNeuralNetwork(
        in_features=args.in_features,
        out_features=args.out_features,
        activation=args.activation,
        dropout=args.dropout,
        bias=args.bias,
    )

    assert network(torch.randn(batch_size, block_size, args.in_features)).size() == (
        batch_size,
        block_size,
        args.in_features,
    ), "Network output size is not correct".capitalize()

    draw_graph(
        model=network, input_data=torch.randn(batch_size, block_size, args.in_features)
    ).visual_graph.render(
        filename=os.path.join(
            config()["path"]["FILES_PATH"], "feedforward_network.png"
        ),
        format="png",
    )
