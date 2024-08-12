import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


class LayerNormalization(nn.Module):
    def __init__(
        self, normalized_shape: int = 512, eps: float = 1e-05, bias: bool = True
    ):
        super(LayerNormalization, self).__init__()

        self.normalized_shape = normalized_shape
        self.epsilon = eps
        self.bias = bias

        self.gamma = nn.Parameter(data=torch.ones((1, 1, normalized_shape)))
        self.beta = nn.Parameter(data=torch.zeros((1, 1, normalized_shape)))

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            self.mean = torch.mean(x, dim=-1)
            self.variance = torch.var(x, dim=-1)

            self.mean = self.mean.unsqueeze(-1)
            self.variance = self.variance.unsqueeze(-1)

            normalized = (
                self.gamma * (x - self.mean) / torch.sqrt(self.variance + self.epsilon)
                + self.beta
            )

            return normalized


if __name__ == "__main__":
    layer_norm = LayerNormalization(normalized_shape=512, eps=1e-05, bias=True)

    print(layer_norm(torch.randn(40, 200, 512)).size())
