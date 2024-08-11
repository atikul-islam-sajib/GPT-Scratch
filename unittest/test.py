import sys
import torch
import unittest
import torch.nn as nn

sys.path.append("./src/")

from utils import config
from scaled_dot_product import scaled_dot_product_attention


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = config()["embedding"]["batch_size"]
        self.block_size = config()["embedding"]["block_size"]

        self.dimension = config()["GPT"]["dimension"]
        self.nheads = config()["GPT"]["nheads"]
        self.dropout = config()["GPT"]["dropout"]
        self.bias = config()["GPT"]["bias"]

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


if __name__ == "__main__":
    unittest.main()
