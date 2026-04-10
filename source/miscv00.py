import torch

def give_f(effect_type):
    if effect_type not in ["id", "sqrt"]:
        raise ValueError('effect_type must be "id" or "sqrt".')

    def a(x):
        return x

    def b(x):
        return x ** 0.5
    f = a if effect_type == "id" else b
    return f

# standardize the sample to 0 mean and unit variance
# read the warning on Zuko's documentation page
# https://zuko.readthedocs.io/stable/api/zuko.flows.spline.html#zuko.flows.spline.NSF


def standardize_sample(s, keep=False):
    """
    Standardize sample to zero mean and identity variance
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
    Count the number of parameters in a normalizing flow
    """
    return sum(p.numel() for p in flows.parameters())
