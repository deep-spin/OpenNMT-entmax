import pytest
import torch
from torch.autograd import gradcheck, grad

from onmt.modules.sparse_losses import (
    sparsemax_loss,
    tsallis15_loss,
    SparsemaxLoss,
    Tsallis15Loss,
    SparsemaxBisectLoss,
    TsallisBisectLoss,
)


def test_sparsemax_loss():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)
        gradcheck(sparsemax_loss, (x, y), eps=1e-5)


def test_sparsemax_bisect_loss():

    sb = SparsemaxBisectLoss(n_iter=50)

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)
        gradcheck(sb, (x, y), eps=1e-5)


@pytest.mark.parametrize('alpha', (1.2, 1.5, 1.75))
def test_tsallis_bisect_loss(alpha):

    ts = TsallisBisectLoss(alpha=alpha, n_iter=50)

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)
        gradcheck(ts, (x, y), eps=1e-5)


def test_tsallis_loss():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)
        gradcheck(tsallis15_loss, (x, y), eps=1e-5)


@pytest.mark.parametrize('Loss', (
    SparsemaxLoss,
    Tsallis15Loss,
    SparsemaxBisectLoss,
    TsallisBisectLoss))
def test_index_ignored(Loss):

    x = torch.randn(20, 6, dtype=torch.float64, requires_grad=True)
    _, y = torch.max(torch.randn_like(x), dim=1)

    loss_ignore = Loss(reduction='sum', ignore_index=y[0])
    loss_noignore = Loss(reduction='sum', ignore_index=-100)

    assert loss_ignore(x, y) < loss_noignore(x, y)
    gradcheck(loss_ignore, (x, y), eps=1e-5)
    gradcheck(loss_noignore, (x, y), eps=1e-5)


if __name__ == '__main__':
    test_sparsemax_loss()
    test_tsallis_loss()
    test_sparsemax_bisect_loss()
    test_tsallis_bisect_loss()
