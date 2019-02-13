import torch
from torch.autograd import gradcheck, grad

from onmt.modules.sparse_losses_vlad import (
    sparsemax_vlad_loss,
    tsallis15_vlad_loss,
    SparsemaxVladLoss
)

from onmt.modules.sparse_losses import SparsemaxLoss


def test_sparsemax_loss():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)

        print(gradcheck(sparsemax_vlad_loss, (x, y), eps=1e-5))


def test_tsallis_loss():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)

        print(gradcheck(tsallis15_vlad_loss, (x, y), eps=1e-5))


def test_builtin_sparsemax_loss():
    for reduction in ('sum', 'elementwise_mean'):
        for ignore_ix in (False, True):
            for _ in range(10):

                x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
                _, y = torch.max(torch.randn_like(x), dim=1)

                # gradcheck fails automatically
                # print(gradcheck(SparsemaxLoss(reduction='sum'), (x, y), eps=1e-5))

                iix = y[0] if ignore_ix else -100
                f = SparsemaxLoss(reduction=reduction, ignore_index=iix)(x, y)
                g = grad(f, x)[0]

                # correct one
                ff = SparsemaxVladLoss(reduction=reduction, ignore_index=iix)(x, y)
                gg = grad(ff, x)[0]

                print("reduction={} ignore_ix={} error={}"
                      .format(reduction, ignore_ix, torch.sum((g - gg) ** 2)))


if __name__ == '__main__':
    test_sparsemax_loss()
    test_tsallis_loss()

    test_builtin_sparsemax_loss()



