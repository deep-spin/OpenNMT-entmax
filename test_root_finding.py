import torch
from torch.autograd import gradcheck

from onmt.modules.root_finding import (
    sparsemax_bisect_loss,
    tsallis_bisect_loss,
)


def test_sparsemax_loss():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)

        print(gradcheck(sparsemax_bisect_loss, (x, y, 50), eps=1e-5))


def test_tsallis_loss(alpha=1.5):

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)

        print(gradcheck(tsallis_bisect_loss, (x, y, alpha, 50), eps=1e-5))


if __name__ == '__main__':
    test_sparsemax_loss()
    for alpha in (1.2, 1.5, 1.7):
        test_tsallis_loss(alpha)



