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


if __name__ == "__main__":
    unittest.main()
