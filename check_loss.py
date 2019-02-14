import time
import numpy as np
import torch

from onmt.modules.sparse_losses import (
    SparsemaxLossFunction,
    SparsemaxLoss,
    Tsallis15Loss,
)

from onmt.modules.sparse_losses_vlad import (
    SparsemaxLoss as SparsemaxVladLoss,
    Tsallis15Loss as Tsallis15VladLoss,
    sparsemax_bisect_loss,
    SparsemaxBisectLoss,
    TsallisBisectLoss,
)

if __name__ == '__main__':
    x = torch.load('test_input.pt', map_location='cpu')
    y = torch.load('test_target.pt', map_location='cpu')
    print("x", x.shape, x.device, x.dtype)
    print("y", y.shape, y.device, y.dtype)

    x_ = x[:10]
    y_ = y[:10]

    ls = SparsemaxLossFunction.apply(x_, y_)
    print(ls)

    ls2 = sparsemax_bisect_loss(x_, y_)
    print(ls2)

    print('--')
    print()

    print("sum reduction")
    ls = SparsemaxLoss(reduction='sum', ignore_index=1)(x, y)
    print("old, ignore_ix   ", ls)
    ls2 = SparsemaxBisectLoss(n_iter=30, reduction='sum', ignore_index=1)(x, y)
    print("bisect, ignore_ix", ls2)
    ls3 = SparsemaxVladLoss(reduction='sum', ignore_index=1)(x, y)
    print("new, ignore_ix   ", ls3)

    ls = Tsallis15Loss(weights='sum', ignore_index=-1)(x, y)
    print("tsallis old no_ignore", ls)
    ls2 = TsallisBisectLoss(n_iter=30, reduction='sum', ignore_index=-1)(x, y)
    print("tsallis bis no_ignore", ls2)
    ls3 = Tsallis15VladLoss(reduction='sum', ignore_index=-1)(x, y)
    print("tsallis new no_ignore", ls3)

    ls = Tsallis15Loss(weights='sum', ignore_index=1)(x, y)
    print("tsallis old    ignore", ls)
    ls2 = TsallisBisectLoss(n_iter=30, reduction='sum', ignore_index=1)(x, y)
    print("tsallis bisect ignore", ls2)
    ls3 = Tsallis15VladLoss(reduction='sum', ignore_index=1)(x, y)
    print("tsallis new    ignore", ls3)

    print()
    print("average reduction")
    ls = Tsallis15Loss(weights='average', ignore_index=-1)(x, y)
    print("tsallis old no_ignore", ls)
    ls2 = TsallisBisectLoss(n_iter=30, reduction='elementwise_mean', ignore_index=-1)(x, y)
    print("tsallis bis no_ignore", ls2)
    ls3 = Tsallis15VladLoss(reduction='elementwise_mean', ignore_index=-1)(x, y)
    print("tsallis new no_ignore", ls3)
