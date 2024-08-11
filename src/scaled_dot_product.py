import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
):
    if (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        assert (
            query.size() == key.size() == value.size()
        ), "query, key, and value must have the same size"

        result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(value.size(-1))

        if mask is not None:
            lookup = torch.triu(
                input=torch.ones_like(mask.unsqueeze(1).unsqueeze(2)), diagonal=1
            )
            lookup = torch.where(lookup == 1.0, 1e-19, lookup)
            result = torch.add(result, lookup)

        attention = torch.softmax(result, dim=-1)
        attention = torch.matmul(attention, value)

        return attention

    else:
        raise ValueError("query, key, and value must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot product attention for GPT".title()
    )
    args = parser.parse_args()

    batch_size = config()["embedding"]["batch_size"]
    block_size = config()["embedding"]["block_size"]

    nheads = config()["GPT"]["nheads"]
    dimension = config()["GPT"]["dimension"]

    scaled = scaled_dot_product_attention(
        query=torch.randn(batch_size, nheads, block_size, dimension // nheads),
        key=torch.randn(batch_size, nheads, block_size, dimension // nheads),
        value=torch.randn(batch_size, nheads, block_size, dimension // nheads),
        mask=torch.randn(batch_size, block_size),
    )

    assert scaled.size() == (
        batch_size,
        nheads,
        block_size,
        dimension // nheads,
    ), "Error in scaled dot product attention".capitalize()
