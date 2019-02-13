import time
import numpy as np
import torch

from onmt.modules.sparse_losses import (
    SparsemaxLossFunction,
    SparsemaxLoss,
    Tsallis15Loss,
)

from onmt.modules.sparse_losses_vlad import (
    SparsemaxVladLoss,
    Tsallis15VladLoss,
)

from onmt.modules.root_finding import (
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

    ls = SparsemaxLoss(reduction='sum', ignore_index=1)(x, y)
    print(ls)
    ls2 = SparsemaxBisectLoss(n_iter=30, reduction='sum', ignore_index=1)(x, y)
    print(ls2)
    ls3 = SparsemaxVladLoss(reduction='sum', ignore_index=1)(x, y)
    print(ls3)

    ls = Tsallis15Loss(weights='sum', ignore_index=-1)(x, y)
    print(ls)
    ls2 = TsallisBisectLoss(n_iter=30, reduction='sum', ignore_index=-1)(x, y)
    print(ls2)
    ls3 = Tsallis15VladLoss(reduction='sum', ignore_index=-1)(x, y)
    print(ls3)

    ls = Tsallis15Loss(weights='sum', ignore_index=1)(x, y)
    print(ls)
    ls2 = TsallisBisectLoss(n_iter=30, reduction='sum', ignore_index=1)(x, y)
    print(ls2)
    ls3 = Tsallis15VladLoss(reduction='sum', ignore_index=1)(x, y)
    print(ls3)
