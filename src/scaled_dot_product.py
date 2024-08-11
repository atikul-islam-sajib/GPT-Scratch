import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")


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
    scaled = scaled_dot_product_attention(
        query=torch.randn(40, 8, 200, 64),
        key=torch.randn(40, 8, 200, 64),
        value=torch.randn(40, 8, 200, 64),
        mask=torch.randn(40, 200),
    )

    print(scaled.size())
