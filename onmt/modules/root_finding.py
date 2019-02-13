"""root finding methods"""
import torch
from torch.autograd import Function

def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)

def _tsallis_gp(x, alpha):
    return (x ** (alpha - 1) - 1) / (alpha - 1)


def _tsallis_gp_inv(y, alpha):
    return (1 + (alpha - 1) * y) ** (1 / (alpha - 1))


def _tsallis_p(X, min_val, alpha):
    X_thr = torch.clamp(X, min=min_val)
    return _tsallis_gp_inv(X_thr, alpha)


class SecantFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=1.5, dim=0, n_iter=500, tol=1e-8):
        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        # max_val, _ = X.max(dim=dim, keepdim=True)
        max_val, _ = X.max(dim=dim)

        gp_zero = _tsallis_gp(0, alpha)
        lo = max_val - _tsallis_gp(1, alpha)
        hi = max_val - _tsallis_gp(1 / d, alpha)

        p_lo = _tsallis_p(X - lo.unsqueeze(dim),
                          min_val=gp_zero, alpha=alpha)
        p_hi = _tsallis_p(X - hi.unsqueeze(dim),
                          min_val=gp_zero, alpha=alpha)
        f_lo = p_lo.sum(dim) - 1
        f_hi = p_hi.sum(dim) - 1

        for it in range(n_iter):

            mask = ((f_lo - f_hi) ** 2) >= tol

            if not mask.any():
                break

            tau = lo[mask] * f_hi[mask]
            tau -= hi[mask] * f_lo[mask]
            tau /= f_hi[mask] - f_lo[mask]

            lo = hi.clone()

            hi[mask] = tau

            _roll_last(p_hi, dim)[mask] = \
                _tsallis_p(_roll_last(X, dim)[mask] - tau.unsqueeze(-1),
                           min_val=gp_zero,
                           alpha=alpha)
            f_lo = f_hi
            f_hi = p_hi.sum(dim) - 1

        # TODO: if not converged, we can force-normalize
        # the entries with mask != 0
        # Should be sth like p_hi[mask] /= p_hi[mask].sum(dim)

        ctx.save_for_backward(p_hi)
        return p_hi

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y ** (2 - ctx.alpha)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None, None, None


_secant = SecantFunction.apply


class TsallisSecant(torch.nn.Module):

    def __init__(self, alpha=1.5, dim=0, n_iter=100, tol=1e-5):
        self.alpha = alpha
        self.dim = dim
        self.n_iter = n_iter
        self.tol = tol

        super(TsallisSecant, self).__init__()

    def forward(self, X):
        return _secant(X, self.alpha, self.dim, self.n_iter, self.tol)


if __name__ == '__main__':
    from tsallis15 import Tsallis15
    torch.manual_seed(52)
    X = torch.randn(2, 3, 4)

    for dim in range(3):
        print(Tsallis15(dim=dim)(X))
        print(TsallisSecant(dim=dim)(X))
