import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from multihead_attention import MultiHeadAttentionLayer
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
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.dimension = dimension
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.eps = eps
