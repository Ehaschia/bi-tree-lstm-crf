import numpy
import torch

def logsumexp(x, dim=None):
    """

    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.

    Returns: The result of the log(sum(exp(...))) operation.

    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + numpy.log(torch.exp(x - xmax).sum())
    else:
        if x.size()[dim] == 1:
            return x.squeeze(dim)
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))