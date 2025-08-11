import math

import torch


__JOINT1_OFFSETS = torch.tensor((math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, 3 * math.pi / 4, - 3 * math.pi / 4)).unsqueeze(0)


def convert_cpg_to_cartesian(
    r: torch.Tensor,
    phi: torch.Tensor,
    l: float,
    h: float,
    g_p: float,
    g_c: float,
    s: float,
    leg_idx: int,
    device: str,
):
    X: torch.Tensor = (
        s * torch.cos(__JOINT1_OFFSETS) + l * (r - 1) * torch.cos(phi)
    )
    Y: torch.Tensor = (
        s * torch.sin(math.pi / 4 * (leg_idx + 1)) + l * (r - 1) * torch.cos(phi)
    )

    sin_phi = torch.sin(phi)
    Z: torch.Tensor = torch.where(sin_phi > 0, -h + g_c * sin_phi, -h + g_p * sin_phi)

    return X, Y, Z
