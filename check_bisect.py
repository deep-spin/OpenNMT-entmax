import torch

from onmt.modules.sparse_activations import Sparsemax
from onmt.modules.root_finding import sparsemax_bisect

sparsemax = Sparsemax(dim=1)

# torch.set_printoptions(precision=2)

if __name__ == '__main__':

    torch.manual_seed(52)
    x = 0.5 * torch.randn(100, 30000, dtype=torch.float32)
    p1 = sparsemax(x.clone())
    p2 = sparsemax_bisect(x)

    print(torch.sum((p1 - p2) ** 2))
