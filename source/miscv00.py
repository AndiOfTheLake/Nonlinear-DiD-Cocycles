"""Miscellaneous module. Mostly helper functions."""

import torch

def give_f(effect_type):
    """Defines a dose-response curve.

    Parameters
    ----------
    effect_type : str
        Effect type. Needs to be one of the values in the list `["id", "sqrt"]`.

    Returns
    -------
    f: function
        The dose response curve. 

    Raises
    ------
    ValueError
        Only two effect types are allowed, `"id"` (the identity function) 
        and `"sqrt"` (the square root function). 
    """
    if effect_type not in ["id", "sqrt"]:
        raise ValueError('effect_type must be "id" or "sqrt".')

    def a(x):
        return x

    def b(x):
        return x ** 0.5
    f = a if effect_type == "id" else b
    return f


def standardize_sample(s, keep=False):
    """Standardizes sample to zero mean and identity variance. 
    Read more about the importance of standardization on Zuko's documentation page: 
    https://zuko.readthedocs.io/stable/api/zuko.flows.spline.html#zuko.flows.spline.NSF

    Parameters
    ----------
    s : torch.Tensor
        The sample to be standardized. 
    keep : bool, optional
        Whether to keep the location (sample mean) and 
        scale (lower Cholesky factor of the sample covariance matrix) parameters 
        of the standardizing transformation, by default False.

    Returns
    -------
    torch.Tensor or dict
        Depends on `keep`.

        If `keep=False`: returns the standardized sample. 

        If `keep=True`: returns a dictionary with the following keys: 
            - "standardized sample": torch.Tensor.
            - "loc": torch.Tensor; the location parameter.
            - "scale": torch.Tensor; the scale parameter.
    """
    loc = s.mean(dim=-2)
    scale = torch.linalg.cholesky(torch.cov(s.T))
    s_std = (scale.inverse() @ (s - loc).unsqueeze(-1)).squeeze(-1)
    if keep:
        return {
            'standardized sample': s_std,
            'loc': loc,
            'scale': scale
        }
    else:
        return s_std


def count_param(flows):
    """
    Counts the number of parameters in a normalizing flow.
    """
    return sum(p.numel() for p in flows.parameters())
