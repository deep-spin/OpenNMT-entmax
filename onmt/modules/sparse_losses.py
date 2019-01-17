import torch
import torch.nn as nn
from torch.autograd import Function
from onmt.modules.sparse_activations import _threshold_and_support, tsallis15
from onmt.utils.misc import aeq


class SparsemaxLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        input_batch, classes = input.size()
        target_batch = target.size(0)
        aeq(input_batch, target_batch)

        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = _threshold_and_support(input, dim=1)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)
        # clamping necessary because of numerical errors: loss should be lower
        # bounded by zero, but negative values near zero are possible without
        # the clamp
        return torch.clamp(x / 2 - z_k + 0.5, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


sparsemax_loss = SparsemaxLossFunction.apply


class SparsemaxLoss(nn.Module):
    """
    An implementation of sparsemax loss, first proposed in "From Softmax to
    Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    (Martins & Astudillo, 2016: https://arxiv.org/pdf/1602.02068). If using
    a sparse output layer, it is not possible to use negative log likelihood
    because the loss is infinite in the case the target is assigned zero
    probability. Inputs to SparsemaxLoss are arbitrary dense real-valued
    vectors (like in nn.CrossEntropyLoss), not probability vectors (like in
    nn.NLLLoss).
    """

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss


class ConjugateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, grad, Omega):
        ctx.save_for_backward(grad)
        return torch.sum(theta * grad, dim=1) - Omega(grad)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output.view(-1, 1), None, None


class FYLoss(torch.nn.Module):

    def __init__(self, weights="average"):
        self.weights = weights
        super(FYLoss, self).__init__()

    def forward(self, theta, y_true):
        self.y_pred = self.predict(theta)
        ret = ConjugateFunction.apply(theta, self.y_pred, self.Omega)

        if len(y_true.shape) == 2:
            # y_true contains label proportions
            ret += self.Omega(y_true)
            ret -= torch.sum(y_true * theta, dim=1)

        elif len(y_true.shape) == 1:
            # y_true contains label integers (0, ..., n_classes-1)

            if y_true.dtype != torch.long:
                raise ValueError("y_true should contains long integers.")

            all_rows = torch.arange(y_true.shape[0])
            ret -= theta[all_rows, y_true]

        else:
            raise ValueError("Invalid shape for y_true.")

        if self.weights == "average":
            return torch.mean(ret)
        else:
            return torch.sum(ret)


class Tsallis15Loss(FYLoss):

    def predict(self, theta):
        return tsallis15(theta, 1)

    def Omega(self, p):
        return (4 / 3) * (torch.sum((p ** 1.5), dim=1) - 1)
