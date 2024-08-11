import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation

        if self.activation == "relu":
            self.activation_fn = nn.ReLU(inplace=True)

        elif self.activation == "gelu":
            self.activation_fn = nn.GELU()

        elif self.activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.layers = list()

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            pass
