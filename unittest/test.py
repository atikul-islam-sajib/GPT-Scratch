import sys
import torch
import unittest
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("./src/")

from utils import config
from scaled_dot_product import scaled_dot_product_attention
from multihead_attention import MultiHeadAttentionLayer
from feedforward_network import FeedForwardNeuralNetwork
from transformer import TransformerEncoderBlock
from positional_encoding import PositionalEncoding
from layer_normalization import LayerNormalization


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = config()["embedding"]["batch_size"]
        self.block_size = config()["embedding"]["block_size"]

        self.dimension = config()["GPT"]["dimension"]
        self.nheads = config()["GPT"]["nheads"]
        self.dropout = config()["GPT"]["dropout"]
        self.bias = config()["GPT"]["bias"]
        self.activation = config()["GPT"]["activation"]
        self.dim_feedforward = config()["GPT"]["dim_feedforward"]
        self.eps = float(config()["GPT"]["eps"])

        self.multihead = MultiHeadAttentionLayer(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
            bias=self.bias,
        )

        self.network = FeedForwardNeuralNetwork(
            in_features=self.dimension,
            out_features=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.layernorm = LayerNormalization(
            normalized_shape=self.dimension, eps=self.eps, bias=True
        )

        self.trasformer = TransformerEncoderBlock(
            dimension=self.dimension,
            nheads=self.nheads,
            dropout=self.dropout,
            activation=self.activation,
            dim_feedforward=self.dim_feedforward,
            eps=self.eps,
            bias=self.bias,
        )

        self.positional_encoding = PositionalEncoding(
            sequence_length=self.block_size,
            dimension=self.dimension,
        )

    def test_scaled_dot_prododuct(self):
        self.assertEqual(
            scaled_dot_product_attention(
                query=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                key=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                value=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                mask=torch.randn(self.batch_size, self.block_size),
            ).size(),
            (
                self.batch_size,
                self.nheads,
                self.block_size,
                self.dimension // self.nheads,
            ),
        )

        self.assertEqual(
            scaled_dot_product_attention(
                query=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                key=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                value=torch.randn(
                    self.batch_size,
                    self.nheads,
                    self.block_size,
                    self.dimension // self.nheads,
                ),
                mask=None,
            ).size(),
            (
                self.batch_size,
                self.nheads,
                self.block_size,
                self.dimension // self.nheads,
            ),
        )

    def test_multihead_attention_layer(self):
        X = torch.randint(0, self.block_size, (self.batch_size * 20, self.block_size))
        y = torch.randint(0, self.block_size, (self.batch_size * 20, self.block_size))
        mask = torch.randint(0, 2, (self.batch_size * 20, self.block_size))

        self.assertEqual(X.size(), y.size())

        dataset = TensorDataset(X, mask)

        dataloader = DataLoader(
            dataset=list(zip(dataset, y)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        (independent, mask), dependent = next(iter(dataloader))

        self.assertEqual(independent.size(), dependent.size(), mask.size())

        embedding_layer = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=self.dimension
        )

        embedding = embedding_layer(independent)

        self.assertEqual(
            self.multihead(x=embedding, mask=None).size(),
            (self.batch_size, self.block_size, self.dimension),
        )

        self.assertEqual(
            self.multihead(x=embedding, mask=mask).size(),
            (self.batch_size, self.block_size, self.dimension),
        )

    def test_feedforward_network(self):
        self.assertEqual(
            self.network(
                torch.randn(self.batch_size, self.block_size, self.dimension)
            ).size(),
            (self.batch_size, self.block_size, self.dimension),
        )

    def test_layer_normalization(self):
        self.assertEqual(
            self.layernorm(
                torch.randn(self.batch_size, self.block_size, self.dimension)
            ).size(),
            (self.batch_size, self.block_size, self.dimension),
        )

        self.seq_model = nn.Sequential(self.network, self.layernorm)

        self.assertEqual(
            self.seq_model(
                torch.randn(self.batch_size, self.block_size, self.dimension)
            ).size(),
            (self.batch_size, self.block_size, self.dimension),
        )

    def test_positional_encoding(self):
        self.assertEqual(
            self.positional_encoding(
                torch.randn(self.batch_size, self.block_size, self.dimension)
            ).size(),
            (self.batch_size // self.batch_size, self.block_size, self.dimension),
        )

    def test_transformer_block(self):
        embedding_layer = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=self.dimension
        )

        input_texts = torch.randint(
            0, self.block_size, (self.batch_size * 10, self.block_size)
        )
        target_text = torch.randint(
            0, self.block_size, (self.batch_size * 10, self.block_size)
        )

        dataset = DataLoader(
            dataset=list(zip(input_texts, target_text)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        X, y = next(iter(dataset))

        X = embedding_layer(X)

        self.assertEqual(X.size(), y.size())
        self.assertEqual(input_texts.size(), target_text.size())
        self.assertEqual(X.size(), (self.batch_size, self.block_size, self.dimension))
        self.assertEqual(
            self.trasformer(x=X, mask=None).size(),
            (self.batch_size, self.block_size, self.dimension),
        )


if __name__ == "__main__":
    unittest.main()
