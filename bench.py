import time
import numpy as np
import torch

from onmt.modules.sparse_activations import Sparsemax, Tsallis15
from onmt.modules.sparse_losses_vlad import (
        sparsemax_vlad_loss,
        tsallis15_vlad_loss,
)
from onmt.modules.root_finding import (
    sparsemax_bisect,
    tsallis_bisect,
    sparsemax_bisect_loss,
    tsallis_bisect_loss,
)

# from onmt.modules.root_finding import TsallisSecant, SparsemaxSecant


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

    import sys

    # location = None
    location = 'cuda'
    if len(sys.argv) > 1:
        location = sys.argv[1]

    x = torch.load('test_input.pt', map_location=location)
    y = torch.load('test_target.pt', map_location=location)
    # x = torch.randn(1280, 35820, device=location)

    # x = x[:, :100]

    print("x", x.shape, x.device, x.dtype)
    print("y", y.shape, y.device, y.dtype)

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

    def _sparsemax_bisect():
        return sparsemax_bisect(x, 10)

    def _sparsemax_loss():
        return sparsemax_vlad_loss(x, y)

    def _sparsemax_bisect_loss():
        return sparsemax_bisect_loss(x, y, 10)

    def _tsallis_bisect():
        return tsallis_bisect(x, 1.5, 10)

    def _tsallis_loss():
        return tsallis15_vlad_loss(x, y)

    def _tsallis_bisect_loss():
        return tsallis_bisect_loss(x, y, 1.5, 10)

    def __softmax_loss():
        lsm = torch.nn.LogSoftmax(dim=-1)
        nll = torch.nn.NLLLoss(ignore_index=ignore_index, reduction='sum')

        return nll(lsm(x), y)

    def __sparsemax_loss():
        sp = SparsemaxLoss(ignore_index=ignore_index, reduction='sum')

        return sp(x, y)

    print("tsallis accuracy", torch.max((_tsallis_bisect() - _tsallis()) ** 2))
    print("sparsemax accuracy", torch.max((_sparsemax_bisect() - _sparsemax()) ** 2))

    torch.cuda.synchronize()
    torch.cuda.synchronize()

#    sort_timings = bench(_sort)
#    print("sorting x ", np.percentile(sort_timings, [25, 50, 75]))

#    topk_timings = bench(_topk)
#    print("top 100 x ", np.percentile(topk_timings, [25, 50, 75]))

    softmax_timings = bench(_softmax)
    print("softmax   ", np.percentile(softmax_timings, [25, 50, 75]))

    sparsemax_timings = bench(_sparsemax)
    print("sparsemax ", np.percentile(sparsemax_timings, [25, 50, 75]))

    sparsemax_b_timings = bench(_sparsemax_bisect)
    print("bisect a=2", np.percentile(sparsemax_b_timings, [25, 50, 75]))

    tsallis_timings = bench(_tsallis)
    print("tsallis15 ", np.percentile(tsallis_timings, [25, 50, 75]))

    tsallis_b_timings = bench(_tsallis_bisect)
    print("bis a=1.5 ", np.percentile(tsallis_b_timings, [25, 50, 75]))

    sp_loss_timings = bench(_sparsemax_loss)
    print("loss   a=2", np.percentile(sp_loss_timings, [25, 50, 75]))

    sp_loss_timings = bench(_sparsemax_bisect_loss)
    print("ls bis a=2", np.percentile(sp_loss_timings, [25, 50, 75]))

    ts_loss_timings = bench(_tsallis_loss)
    print("loss a=1.5", np.percentile(ts_loss_timings, [25, 50, 75]))

    ts_loss_timings = bench(_tsallis_bisect_loss)
    print("lsbs a=1.5", np.percentile(ts_loss_timings, [25, 50, 75]))


