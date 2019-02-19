import torch
from onmt.modules.sparse_losses import SparsemaxTopKLoss

if __name__ == '__main__':

    n_samples = 1024
    n_hid = 512
    n_vocab = 36000

    X = torch.randn(n_samples, n_hid)
    y = torch.empty(n_samples, dtype=torch.long).random_(0, n_vocab)

    out_layer = torch.nn.Linear(n_hid, n_vocab, bias=True)
    Z = out_layer(X)

    ignore_ix, _ = y.mode()
    Z_ignored = Z[y != ignore_ix]
    y_ignored = y[y != ignore_ix]

    print(Z_ignored.shape, y_ignored.shape)


    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=ignore_ix, reduction='sum')
    loss = loss_fun(Z, y)
    print(loss)

    # try the naive way
    loss_fun_noignore = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss_fun_noignore(Z_ignored, y_ignored)
    print(loss)

    # sparsemax
    loss_fun = SparsemaxTopKLoss(ignore_index=ignore_ix, reduction='sum')
    loss = loss_fun(Z, y)
    print(loss)

    # sparsemax naive
    loss_fun_noignore = SparsemaxTopKLoss(reduction='sum')
    loss = loss_fun_noignore(Z_ignored, y_ignored)
    print(loss)
