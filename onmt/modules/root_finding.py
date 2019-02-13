# bisection
import torch
from torch.autograd import Function
from onmt.utils.misc import aeq as assert_equal  # HORRIBLE NAMING!


def _p(x, tau):
    return torch.clamp(x - tau, min=0)


def _tsallis_gp_true(x, alpha):
    return (x ** (alpha - 1) - 1) / (alpha - 1)


def _tsallis_gp_inv_true(y, alpha):
    return (1 + (alpha - 1) * y) ** (1 / (alpha - 1))


def _tsallis_p_true(X, min_val, alpha):
    X_thr = torch.clamp(X, min=min_val)
    return _tsallis_gp_inv(X_thr, alpha)


def _tsallis_gp(x, alpha):
    return x ** (alpha - 1)


def _tsallis_gp_inv(y, alpha):
    return y ** (1 / (alpha - 1))


def _tsallis_p(X,  alpha):
    return _tsallis_gp_inv(torch.clamp(X, min=0), alpha)


class SparsemaxBisectFunction(Function):

    @staticmethod
    def forward(ctx, X, n_iter=15):

        dim = 1
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - 1
        tau_hi = max_val - (1 / d)

        f_lo = _p(X, tau_lo).sum(dim) - 1
        f_hi = _p(X, tau_hi).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = _p(X, tau_m)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        ctx.save_for_backward(p_m)
        return p_m

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None


class TsallisBisectFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=1.5, n_iter=15):

        ctx.alpha = alpha
        dim = 1
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        minv = _tsallis_gp(0, alpha)

        tau_lo = max_val - _tsallis_gp(1, alpha)
        tau_hi = max_val - _tsallis_gp(1 / d, alpha)

        f_lo = _tsallis_p(X - tau_lo, alpha).sum(dim) - 1
        f_hi = _tsallis_p(X - tau_hi, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = _tsallis_p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        ctx.save_for_backward(p_m)
        return p_m

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y ** (2 - ctx.alpha)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


class SparsemaxBisectLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, n_iter=25):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = sparsemax_bisect(input, n_iter)

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
        return grad, None, None


class SparsemaxVladLossFunction(SparsemaxBisectLossFunction):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = sparsemax(input, n_iter)

        loss = -(p_star ** 2).sum(dim=1)
        loss += 1
        loss /= 2

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss


class TsallisBisectLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, alpha=1.5, n_iter=15):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = tsallis_bisect(input, alpha, n_iter)

        loss = -(p_star ** alpha).sum(dim=1)
        loss += 1
        loss /= alpha * (alpha - 1)

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


class Tsallis15VladLossFunction(TsallisBisectLossFunction):

    @staticmethod
    def forward(ctx, input, target, alpha=1.5):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = tsallis_bisect(input, alpha, n_iter)

        loss = -(p_star ** alpha).sum(dim=1)
        loss += 1
        loss /= alpha * (alpha - 1)

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


class _GenericLoss(torch.nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, input, target):
        loss = self.loss(input, target)
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


class SparsemaxBisectLoss(_GenericLoss):

    def __init__(self, n_iter=15, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.n_iter = n_iter
        super(SparsemaxBisectLoss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return sparsemax_bisect_loss(input, target, self.n_iter)


class TsallisBisectLoss(_GenericLoss):

    def __init__(self, alpha=1.5, n_iter=15, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.alpha = alpha
        self.n_iter = n_iter
        super(TsallisBisectLoss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return tsallis_bisect_loss(input, target, self.alpha, self.n_iter)


sparsemax_bisect = SparsemaxBisectFunction.apply
tsallis_bisect = TsallisBisectFunction.apply
sparsemax_bisect_loss = SparsemaxBisectLossFunction.apply
tsallis_bisect_loss = TsallisBisectLossFunction.apply
