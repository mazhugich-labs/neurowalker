import torch


def calc_psi(default_alpha: torch.Tensor) -> torch.Tensor:
    """Calculate inter-oscillator phase offset matrix"""
    return default_alpha.T - default_alpha


def calc_m(
    psi: torch.Tensor,
    self_weight: float,
    in_group_weight: float,
    of_group_weight: float,
    threshold: float,
    device: str,
) -> torch.Tensor:
    """Calculate coupling weight matrix based on the inter-oscillator phase offset matrix"""

    weights = torch.full_like(psi, of_group_weight, device=device)
    weights[(psi >= -threshold) & (psi <= threshold)] = in_group_weight
    weights[range(psi.shape[0]), range(psi.shape[0])] = self_weight

    return weights
