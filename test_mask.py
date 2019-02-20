import torch
import pytest

from onmt.modules.sparse_activations import (
    Sparsemax,
    Tsallis15,
    SparsemaxTopK,
    Tsallis15TopK,
)

from onmt.modules.root_finding import (
    sparsemax_bisect,
    tsallis_bisect,
)

funcs = [
    Sparsemax(dim=1),
    Tsallis15(dim=1),
    SparsemaxTopK(dim=1),
    Tsallis15TopK(dim=1),
    sparsemax_bisect,
    tsallis_bisect,
]


@pytest.mark.parametrize('func', funcs)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_mask(func, dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 6, dtype=dtype)
    x[:, 3:] = -float('inf')
    x0 = x[:, :3]

    y = func(x)
    y0 = func(x0)

    y[:, :3] -= y0

    assert torch.allclose(y, torch.zeros_like(y))

