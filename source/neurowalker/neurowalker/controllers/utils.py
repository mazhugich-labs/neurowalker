import torch


def generate_cartesian_from_cpg_mp(
    cpg_state: dict[str, torch.Tensor],
    mp_state: dict[str, torch.Tensor],
    l1_rot: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x = mp_state["d"] * (cpg_state["r"] - 1) * torch.cos(cpg_state["phi"]) * torch.cos(
        cpg_state["omega"]
    ) + mp_state["s"] * torch.cos(l1_rot)
    y = mp_state["d"] * (cpg_state["r"] - 1) * torch.cos(cpg_state["phi"]) * torch.sin(
        cpg_state["omega"]
    ) + mp_state["s"] * torch.sin(l1_rot)
    sin_term = torch.sin(cpg_state["phi"])
    z = (
        -mp_state["h"]
        + torch.where(sin_term > 0, mp_state["g_c"], mp_state["g_p"]) * sin_term
    )

    return {"x": x, "y": y, "z": z}
