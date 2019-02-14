import pytest
import torch
from torch.autograd import gradcheck, grad
from functools import partial

from onmt.modules.sparse_losses import (
    SparsemaxLoss,
    SparsemaxTopKLoss,
    Tsallis15Loss,
    Tsallis15TopKLoss,
    SparsemaxBisectLoss,
    TsallisBisectLoss,
)


# make data
Xs = [torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
        for _ in range(10)]

ys = [torch.max(torch.randn_like(X), dim=1)[1] for X in Xs]


losses = [
    SparsemaxLoss,
    partial(SparsemaxTopKLoss, k=5),
    Tsallis15Loss,
    partial(Tsallis15TopKLoss, k=5),
    SparsemaxBisectLoss,
    TsallisBisectLoss,
]


@pytest.mark.parametrize('Loss', losses)
def test_non_neg(Loss):

    for X, y in zip(Xs, ys):
        ls = Loss(reduction='none')
        lval = ls(X, y)
        assert torch.all(lval >= 0)


@pytest.mark.parametrize('Loss', losses)
@pytest.mark.parametrize('ignore_index', (False, True))
@pytest.mark.parametrize('reduction', ('sum', 'elementwise_mean'))
def test_loss(Loss, ignore_index, reduction):

    for X, y in zip(Xs, ys):
        iix = y[0] if ignore_index else -100
        ls = Loss(ignore_index=iix, reduction=reduction)
        gradcheck(ls, (X, y), eps=1e-5)


@pytest.mark.parametrize('Loss', losses)
def test_index_ignored(Loss):

    x = torch.randn(20, 6, dtype=torch.float64, requires_grad=True)
    _, y = torch.max(torch.randn_like(x), dim=1)

    loss_ignore = Loss(reduction='sum', ignore_index=y[0])
    loss_noignore = Loss(reduction='sum', ignore_index=-100)

    assert loss_ignore(x, y) < loss_noignore(x, y)


if __name__ == '__main__':
    test_sparsemax_loss()
    test_tsallis_loss()
    test_sparsemax_bisect_loss()
    test_tsallis_bisect_loss()
