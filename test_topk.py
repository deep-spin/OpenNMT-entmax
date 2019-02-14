import pytest
import torch
from torch.autograd import gradcheck


from onmt.modules.sparse_activations import (
    _threshold_and_support,
    _threshold_and_support_topk,
    _tsallis_threshold_and_support,
    _tsallis_threshold_and_support_topk,
    SparsemaxTopK,
    Tsallis15TopK,
)

from onmt.modules.sparse_losses import (
    SparsemaxLoss,
    SparsemaxTopKLoss,
    Tsallis15Loss,
    Tsallis15TopKLoss,
)


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('Map', (SparsemaxTopK, Tsallis15TopK))
def test_mapping(dim, Map):
    f = Map(dim=dim, k=3)

    for _ in range(10):
        x = torch.randn(5, 6, 7, requires_grad=True, dtype=torch.float64)
        gradcheck(f, (x,))


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('coef', (0.00001, 0.5, 10000))
def test_tsallis_topk(dim, coef):
    x = coef * torch.randn(10, 11, 12)
    tau1, supp1 = _tsallis_threshold_and_support(x, dim=dim)
    tau2, supp2 = _tsallis_threshold_and_support_topk(x, dim=dim, k=5)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('coef', (0.00001, 0.5, 10000))
@pytest.mark.parametrize('k', (5, 30))
def test_sparsemax_topk(dim, coef, k):

    x = coef * torch.randn(10, 11, 12)
    tau1, supp1 = _threshold_and_support(x, dim=dim)
    tau2, supp2 = _threshold_and_support_topk(x, dim=dim, k=k)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)


def _bench(f):
    timings = []
    for _ in range(10):
        tic = time.perf_counter()
        f()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        timings.append(toc - tic)
    return np.percentile(timings, [25, 50, 75])


def check_speed():
    # device = 'cpu'
    device = 'cuda'
    x = torch.randn(1024, 32000, device=device)
    _, y = torch.max(torch.randn_like(x), dim=1)
    ix = y[0]

    args = dict(reduction='sum', ignore_index=ix)

    sp1 = partial(SparsemaxLoss(**args), input=x, target=y)
    sp2 = partial(SparsemaxTopKLoss(k=500, **args), input=x, target=y)
    ts1 = partial(Tsallis15Loss(**args), input=x, target=y)
    ts2 = partial(Tsallis15TopKLoss(k=500, **args), input=x, target=y)
    # sp1 = partial(_threshold_and_support, input=x, dim=1)
    # sp2 = partial(_threshold_and_support_topk, input=x, dim=1, k=500)

#    ts1 = partial(_tsallis_threshold_and_support, input=x, dim=1)
#    ts2 = partial(_tsallis_threshold_and_support_topk, input=x, dim=1, k=5000)

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    print("sparsemax topk", _bench(sp2))
    print("sparsemax full", _bench(sp1))
    print("tsallis15 topk", _bench(ts2))
    print("tsallis15 full", _bench(ts1))

    print(((sp1() - sp2()) ** 2).sum())
    print(((ts1() - ts2()) ** 2).sum())


if __name__ == '__main__':
     import numpy as np
     from functools import partial
     import time
     check_speed()
