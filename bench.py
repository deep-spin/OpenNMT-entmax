import time
import numpy as np
import torch

from onmt.modules.sparse_activations import Sparsemax, Tsallis15
from onmt.modules.sparse_losses import SparsemaxLoss, Tsallis15Loss


def bench(f):
    timings = []
    for _ in range(10):
        tic = time.perf_counter()
        f()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        timings.append(toc - tic)
    return np.array(timings)


if __name__ == '__main__':

    ignore_index = 1
    # y = torch.load('test_target.pt')

    import sys

    location = None
    if len(sys.argv) > 1:
        location = sys.argv[1]

    x = torch.load('test_input.pt', map_location=location)

    # x.detach_()
    # y.detach_()

    # x = x[:128]
    # y = y[:128]

    # x = x[:, :2000]
    print("x shape", x.shape, x.device, x.dtype)
    # print("y shape", y.shape, y.device)

    def _sort():
        return torch.sort(x, dim=-1)

    def _topk():
        return torch.topk(x, 100, dim=-1)

    def _softmax():
        return torch.nn.Softmax(dim=-1)(x)

    def _sparsemax():
        return Sparsemax(dim=-1)(x)

    def _tsallis():
        return Tsallis15(dim=-1)(x)

    def softmax_loss():
        lsm = torch.nn.LogSoftmax(dim=-1)
        nll = torch.nn.NLLLoss(ignore_index=ignore_index, reduction='sum')

        return nll(lsm(x), y)

    def sparsemax_loss():
        sp = SparsemaxLoss(ignore_index=ignore_index, reduction='sum')

        return sp(x, y)

    torch.cuda.synchronize()
    torch.cuda.synchronize()

    sort_timings = bench(_sort)
    print("sorting x ", np.percentile(sort_timings, [25, 50, 75]))

    topk_timings = bench(_topk)
    print("top 100 x ", np.percentile(topk_timings, [25, 50, 75]))

    softmax_timings = bench(_softmax)
    print("softmax   ", np.percentile(softmax_timings, [25, 50, 75]))

    sparsemax_timings = bench(_sparsemax)
    print("sparsemax ", np.percentile(sparsemax_timings, [25, 50, 75]))

    tsallis_timings = bench(_tsallis)
    print("tsallis15 ", np.percentile(tsallis_timings, [25, 50, 75]))



