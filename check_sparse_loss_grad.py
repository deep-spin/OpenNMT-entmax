import torch
import time
import numpy as np

from onmt.modules.sparse_losses import SparsemaxTopKLoss, _omega_sparsemax
from onmt.modules.sparse_activations import sparsemax_topk


def _bench(f):
    timings = []
    for _ in range(10):
        tic = time.perf_counter()
        f()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        timings.append(toc - tic)
    return np.percentile(timings, [25, 50, 75])


class SparsemaxLossDirect(torch.autograd.Function):

    # assumes sum averaging and no ignore ix
    @staticmethod
    def forward(ctx, X, W, b, y):

        Z = X @ W.t() + b

        P = sparsemax_topk(Z, 1, 512)

        loss = _omega_sparsemax(P).sum()
        loss -= torch.gather(Z, dim=1, index=y.unsqueeze(1)).sum()

        # P.scatter_add_(1, y.unsqueeze(1), torch.full_like(P, -1))

        loss += torch.einsum("ij,ij->", P, Z)

        mask = P != 0
        p_ix = mask.nonzero().t()
        p_data = P[mask]

        rg = torch.arange(len(y), device=y.device)
        y_ix = torch.stack([rg, y])
        y_data = torch.ones_like(y, dtype=Z.dtype)

        ix = torch.cat([p_ix, y_ix], dim=1)
        data = torch.cat([p_data, y_data])
        ctx.save_for_backward(ix, data, X, W, b)
        ctx.shape = P.shape

        return loss

    @staticmethod
    def backward(ctx, dL):
        ix, data, X, W, b = ctx.saved_tensors
        data *= dL
        dZ = torch.sparse_coo_tensor(ix, data, ctx.shape)

        # if coded this way, we have no gains from sparsity!
        # because dZ @ anything will have dense results.
        # This needs a new approach: the resulting dX / dW will be
        # row-sparse / column-sparse, so we can try to build those tensors
        # explicity, by storing only nonzero columns of dZ, and doing
        # dense multiplies.
        dX = dZ @ W
        dW = dZ.t() @ X
        db = dZ.t() @ torch.ones(dZ.shape[0], 1, device=dZ.device)  # .sum(dim=1)
        db = db.squeeze(1)
        dy = None

        return dX, dW, db, dy


if __name__ == '__main__':

    n_samples = 1024
    # n_samples = 500
    n_hid = 512
    n_vocab = 36000

    # torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)

    X = torch.randn(n_samples, n_hid)
    y = torch.empty(n_samples, dtype=torch.long).random_(0, n_vocab)
    out_layer = torch.nn.Linear(n_hid, n_vocab, bias=True)

    cuda = True
    if cuda:
        X = X.cuda()
        y = y.cuda()
        out_layer = out_layer.cuda()

    ignore_ix, _ = y.mode()

    def softmax_old():
        loss_fun = torch.nn.CrossEntropyLoss(ignore_index=ignore_ix, reduction='sum')
        Z = out_layer(X)
        loss = loss_fun(Z, y)
        return loss

    def softmax_naive():
        loss_fun_noignore = torch.nn.CrossEntropyLoss(reduction='sum')
        Z_ignored = out_layer(X[y != ignore_ix])
        y_ignored = y[y != ignore_ix]
        loss = loss_fun_noignore(Z_ignored, y_ignored)
        return loss

    def sparsemax_old():
        loss_fun = SparsemaxTopKLoss(ignore_index=ignore_ix, reduction='sum', k=512)
        Z = out_layer(X)
        loss = loss_fun(Z, y)
        return loss

    def sparsemax_naive():
        loss_fun = SparsemaxTopKLoss(reduction='sum', k=512)
        Z_ignored = out_layer(X[y != ignore_ix])
        y_ignored = y[y != ignore_ix]
        loss = loss_fun(Z_ignored, y_ignored)
        grad = torch.autograd.grad(loss, out_layer.weight)
        return loss, grad

    def sparsemax_direct():
        loss_fun = SparsemaxLossDirect.apply
        X_ignored = X[y != ignore_ix]
        y_ignored = y[y != ignore_ix]
        loss = loss_fun(X_ignored, out_layer.weight, out_layer.bias, y_ignored)
        grad = torch.autograd.grad(loss, out_layer.weight)
        return loss, grad

    print(softmax_old().item())
    print(softmax_naive().item())
    print(sparsemax_old().item())
    print(sparsemax_naive()[0].item())
    print(sparsemax_direct()[0].item())

    #print(torch.autograd.grad(sparsemax_naive(), out_layer.weight))
    #print(torch.autograd.grad(sparsemax_direct(), out_layer.weight))

    torch.cuda.synchronize()
    torch.cuda.synchronize()

    print("softmax old", _bench(softmax_old))
    print("softmax naive", _bench(softmax_naive))
    print("sparsemax old", _bench(sparsemax_old))
    print("sparsemax naive", _bench(sparsemax_naive))
    print("sparsemax direct", _bench(sparsemax_direct))
