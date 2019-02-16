import torch
from torch.autograd import Function
import torch.nn as nn

from onmt.modules.sparse_activations import (
    _make_ix_like,
    sparsemax,
    _threshold_and_support_topk,
    sparsemax_topk)


def sparse_out_sparsemax_1(X, k=2):
    tau, nnz = _threshold_and_support_topk(X, dim=1, k=k)
    support = X > tau
    sparse_ix = support.nonzero().t()
    data = X[support]
    sparse_tau = torch.cat([torch.full((k,), t.item(), device=X.device) for t, k in zip(tau, nnz)])
    return torch.sparse_coo_tensor(sparse_ix, data - sparse_tau, X.shape)


def sparse_out_sparsemax_2(X, k=2):
    tau, nnz = _threshold_and_support_topk(X, dim=1, k=k)
    support = X > tau
    sparse_ix = support.nonzero().t()
    data = X[support] - tau.expand_as(X)[support]
    return torch.sparse_coo_tensor(sparse_ix, data, X.shape)


def sparse_out_approx_sparsemax(X, k=2):
    dim = 1
    n, d = X.shape

    if k >= d:  # do full sort
        X_srt, inv_ix = torch.sort(X, dim=dim, descending=True)
    else:
        X_srt, inv_ix = torch.topk(X, k=k, dim=dim)

    X_cumsum = X_srt.cumsum(dim) - 1
    rhos = _make_ix_like(X_srt, dim)
    support = rhos * X_srt > X_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = X_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    row = support.to(dtype=torch.long) * \
        torch.arange(1, n + 1, dtype=inv_ix.dtype, device=X.device).unsqueeze(1)
    row = row[support] - 1

    sparse_ix = torch.stack((row, inv_ix[support]))
    data = (X_srt - tau)[support]
    return torch.sparse_coo_tensor(sparse_ix, data, X.shape)


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
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    print("vocab={} k={} device={}".format(vocab_size, k, device))
    X = torch.randn(1024, vocab_size, device=device)

    f0 = partial(sparsemax_topk, X, 1, k)
    f1 = partial(sparse_out_sparsemax_1, X=X, k=k)
    f2 = partial(sparse_out_sparsemax_2, X=X, k=k)
    f3 = partial(sparse_out_approx_sparsemax, X=X, k=k)

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    print("sparsemax0", _bench(f0))
    print("sparsemax1", _bench(f1))
    print("sparsemax2", _bench(f2))
    print("sparsemax3", _bench(f3))


if __name__ == '__main__':
    from functools import partial
    import time
    import sys
    import numpy as np
    torch.manual_seed(43)
    X = .5 * torch.randn(4, 6)
    P = sparse_out_sparsemax_2(X)
    print(P)
    print(torch.sum((sparsemax(X, 1) - P) ** 2))
    print()
    check_speed()
