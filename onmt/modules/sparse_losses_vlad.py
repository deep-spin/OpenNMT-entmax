import torch
from torch.autograd import Function
from onmt.utils.misc import aeq as assert_equal  # HORRIBLE NAMING!

from onmt.modules.sparse_activations import sparsemax, tsallis15
from onmt.modules.root_finding import _GenericLoss


class SparsemaxVladLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = sparsemax(input.clone(), 1)

        loss = -(p_star ** 2).sum(dim=1)
        loss += 1
        loss /= 2

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        p_star, = ctx.saved_tensors
        grad = grad_output.unsqueeze(1) * p_star
        return grad, None, None, None


class Tsallis15VladLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, alpha=1.5):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = tsallis15(input.clone(), 1)

        loss = -(p_star ** alpha).sum(dim=1)
        loss += 1
        loss *= 4 / 3

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        p_star, = ctx.saved_tensors
        grad = grad_output.unsqueeze(1) * p_star
        return grad, None, None, None


sparsemax_vlad_loss = SparsemaxVladLossFunction.apply
tsallis15_vlad_loss = Tsallis15VladLossFunction.apply


class SparsemaxVladLoss(_GenericLoss):

    def loss(self, input, target):
        return sparsemax_vlad_loss(input, target)


class Tsallis15VladLoss(_GenericLoss):

    def loss(self, input, target):
        return tsallis15_vlad_loss(input, target)
