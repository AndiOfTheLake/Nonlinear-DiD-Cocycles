""" 
CMMD-U module. 
The functions in this module compute the CMMD-U loss or carry out related tasks. 
"""

import torch
import numpy as np


def subsample(s: torch.tensor, n: int = 6000):
    """Randomly subsamples data to compute median heuristic (to avoid memory overload).

    Parameters
    ----------
    s : torch.Tensor
        Data to be subsapmled from.
    n : int, optional
        Size of the subsample, by default 6000.

    Returns
    -------
    torch.Tensor
        A random subsample of the input data.
    """
    return s[torch.randperm(len(s))[:n], :]


def get_s1(y_hat_batch, y_batch, y_prod_batch, bandwidth):
    """Computes the first sum in CMMD-U (on a minibatch).

    Parameters
    ----------
    y_hat_batch : torch.Tensor
        Estimated outcome using TMI maps (on the given minibatch). 
        All combinations of "to" and "from" states are given in `y_prod_batch` (see below).
    y_batch : torch.Tensor
        Outcome tensor, typically of shape `(B, d)`, where `B` is the batch size and 
        `d` is the outcome dimension (as `d` in `R^d`).
    y_prod_batch : torch.Tensor
        A Cartesian product of the indices of observations of shape `(B^2, 2)`, 
        where `B` is the batch size. The first and second columns indicate 
        the "to" and "from" states of the TMI transport maps.
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).

    Returns
    -------
    s1 : torch.Tensor
        The first sum in CMMD-U.
    """
    y_target = y_batch[y_prod_batch[:, -2], :]
    # indices of non-trivial transformations
    # there should be n trivial transformations, where n is the (mini)batch size
    mask_non_trivial = y_prod_batch[:, -2] != y_prod_batch[:, -1]
    s1 = torch.exp(
        input=-0.5 * (y_hat_batch[mask_non_trivial, :] -
                      y_target[mask_non_trivial, :])
        .pow(2)
        .sum(-1) / (bandwidth ** 2)
    ).sum()
    return s1

# Second term
# This really needs some advanced broadcasting!
# study the broadcasting by ChatGPT.
# Check the definition of median heuristic
# (compare with https://en.wikipedia.org/wiki/Radial_basis_function_kernel)


def get_s2(dt, bandwidth):
    """Computes the second sum in CMMD-U.

    Parameters
    ----------
    dt : torch.Tensor
        A tensor of shape `(B^2, d+2)`, 
        where `B` is the batch size and `d` is the outcome dimension. 
        In practice, `dt=torch.cat((y_hat_batch, y_prod_batch), dim=-1)`.
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).

    Returns
    -------
    torch.Tensor
        The second sum in CMMD-U.
    """
    y_hat = dt[:, :-2]
    i = dt[:, -2].long()
    j = dt[:, -1].long()
    # This needs to be an integer for torch.empty()
    n = int(np.sqrt((y_hat.shape[0])))
    device = dt.device
    dtype = dt.dtype
    # pack into (n, n, d)
    d = y_hat.shape[1]
    Y = torch.empty(n, n, d, device=device, dtype=dtype)
    Y[i, j] = y_hat  # (n, n, 2)
    # all pairwise diffs Y[i,j] - Y[i,k]
    X1 = Y.unsqueeze(-2)  # (n, n, 1, 2); faster than Y[:, :, None, :]
    X2 = Y.unsqueeze(-3)  # (n, 1, n, 2); faster than Y[:, None, :, :]
    # The shape of X1 - X2 is (n, n, n, 2) due to broadcasting
    dist_sq = (X1 - X2).pow(2).sum(-1)  # (n, n, n)
    # Gaussian kernel
    Kvol = torch.exp(-0.5 * dist_sq / (bandwidth ** 2))  # (n, n, n)

    # mask out any index collisions i=j, i=k, or j=k
    idx = torch.arange(n, device=device)
    I = idx.view(n, 1, 1)
    J = idx.view(1, n, 1)
    K = idx.view(1, 1, n)
    mask = (J != I) & (K != I) & (J != K)  # (n, n, n)
    return Kvol[mask].sum()


def compute_cmmdu(y_batch, cond_var_batch, bandwidth, flows):
    """Computes the CMMD-U loss (on a minibatch).

    Parameters
    ----------
    y_batch : torch.Tensor
        Outcome tensor, typically of shape `(B, d)`, where `B` is the batch size and 
        `d` is the outcome dimension (as `d` in `R^d`).
    cond_var_batch : torch.Tensor
        Conditioning variable.
    bandwidth : torch.Tensor
        Bandwidth for the kernel (usually calculated using the median heuristic).
    flows : zuko.flows.NSF
        Neural spline flow to be trained to model the cocycle.

    Returns
    -------
    torch.Tensor
        CMMD-U loss (on that minibatch).
    """
    device = y_batch.device

    y_prod_batch = torch.cartesian_prod(
        torch.arange(y_batch.shape[0], device=device),
        torch.arange(y_batch.shape[0], device=device))

    # "from" and "to" states
    c_from = cond_var_batch[y_prod_batch[:, 1], :]
    c_to = cond_var_batch[y_prod_batch[:, 0], :]
    y_from = y_batch[y_prod_batch[:, 1]]

    # Check if the "from" and "to" states are the same
    # If so, there is no need to compute anything since the transformation is the id
    same_states = (c_from == c_to)
    mask = same_states.all(dim=1)

    y_hat_batch = torch.empty_like(y_from)
    y_hat_batch[mask, :] = y_from[mask, :]
    inv_mask = torch.logical_not(mask)
    y_hat_batch[inv_mask, :] = flows(c=c_to[inv_mask, :]).transform.inv(
        flows(c=c_from[inv_mask, :]).transform(y_from[inv_mask, :])
    )
    s1 = get_s1(y_hat_batch, y_batch, y_prod_batch, bandwidth)
    s2 = get_s2(dt=torch.cat((y_hat_batch, y_prod_batch), dim=-1),
                bandwidth=bandwidth)
    return (s2 / (y_batch.shape[0] - 2) - 2 * s1) / (y_batch.shape[0] - 1)
