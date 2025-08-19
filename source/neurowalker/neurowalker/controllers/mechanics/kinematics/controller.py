"""This module defines Neurowalker Kinematics Controller"""

import math
from dataclasses import dataclass

import torch


# ---------- Configuration ----------
@dataclass(frozen=True)
class SHIFT:
    L1: float = 0.0415
    L2: float = 0.078
    L3_1: float = 0.0515
    L3_2: float = 0.106
    L3: float = 0.1553956692888851


@dataclass
class ROTATION:
    L1: torch.Tensor = torch.tensor(
        (
            (
                math.pi / 4,
                -math.pi / 4,
                math.pi / 2,
                -math.pi / 2,
                3 * math.pi / 4,
                -3 * math.pi / 4,
            ),
        )
    )
    L1_COS_TERM: torch.Tensor = torch.tensor(
        (
            (
                math.pi / 4,
                -math.pi / 4,
                math.pi / 2,
                -math.pi / 2,
                3 * math.pi / 4,
                -3 * math.pi / 4,
            ),
        )
    ).cos()
    L1_SIN_TERM: torch.Tensor = torch.tensor(
        (
            (
                math.pi / 4,
                -math.pi / 4,
                math.pi / 2,
                -math.pi / 2,
                3 * math.pi / 4,
                -3 * math.pi / 4,
            ),
        )
    ).sin()
    L3_2: float = -math.pi / 9
    L3: float = -0.23547211171331967


@dataclass
class KinematicsCfg:
    shift: SHIFT = SHIFT()

    rotation: ROTATION = ROTATION()


# ---------- Controller ----------
class KinematicsController:
    def __init__(self, device: str):
        self.cfg = KinematicsCfg()
        self.cfg.rotation.L1 = self.cfg.rotation.L1.to(device)
        self.cfg.rotation.L1_COS_TERM = self.cfg.rotation.L1_COS_TERM.to(device)
        self.cfg.rotation.L1_SIN_TERM = self.cfg.rotation.L1_SIN_TERM.to(device)

    # ---------- Intrinsic methods ----------
    def __solve_foot_position_from_cpg_state(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        heading: torch.Tensor,
        s: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        g_c: torch.Tensor,
        g_p: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute foot position from CPG state"""

        x: torch.Tensor = (
            -d * (r - 1) * torch.cos(phi) * torch.cos(heading)
            + s * self.cfg.rotation.L1_COS_TERM
        )
        y: torch.Tensor = (
            -d * (r - 1) * torch.cos(phi) * torch.sin(heading)
            + s * self.cfg.rotation.L1_SIN_TERM
        )
        sin_term = torch.sin(phi)
        z: torch.Tensor = -h + torch.where(sin_term > 0, g_c, g_p) * sin_term

        return {"x": x, "y": y, "z": z}

    def __solve_joint_position(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute joint position"""

        r = (x**2 + y**2) ** 0.5 - self.cfg.shift.L1
        c = (r**2 + z**2) ** 0.5

        alpha = torch.atan2(z, r)
        beta = torch.acos(
            torch.clamp(
                (self.cfg.shift.L2**2 + c**2 - self.cfg.shift.L3**2)
                / (2 * self.cfg.shift.L2 * c),
                -1,
                1,
            )
        )
        gamma = torch.acos(
            torch.clamp(
                (self.cfg.shift.L2**2 + self.cfg.shift.L3**2 - c**2)
                / (2 * self.cfg.shift.L2 * self.cfg.shift.L3),
                -1,
                1,
            )
        )

        q1 = torch.atan2(y, x) - self.cfg.rotation.L1
        q2 = -alpha - beta
        q3 = gamma - ROTATION.L3 - math.pi

        return {
            "q1": q1,
            "q2": q2,
            "q3": q3,
        }

    # ---------- Public API ----------
    def solve_foot_position_from_cpg_state(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        heading: torch.Tensor,
        s: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        g_c: torch.Tensor,
        g_p: torch.Tensor,
        use_jit: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute foot position from CPG state"""

        if use_jit:
            return solve_foot_position_from_cpg_state_jit(
                r,
                phi,
                heading,
                s,
                h,
                d,
                g_c,
                g_p,
                self.cfg.rotation.L1_COS_TERM,
                self.cfg.rotation.L1_SIN_TERM,
            )

        return self.__solve_foot_position_from_cpg_state(
            r, phi, heading, s, h, d, g_c, g_p
        )

    def solve_joint_position(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        use_jit: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute joint position"""

        if use_jit:
            return solve_joint_position_jit(
                x,
                y,
                z,
                self.cfg.shift.L1,
                self.cfg.shift.L2,
                self.cfg.shift.L3,
                self.cfg.rotation.L1,
                self.cfg.rotation.L3,
            )

        return self.__solve_joint_position(x, y, z)


# ---------- PyTorch JIT scripts ----------
@torch.jit.script
def solve_foot_position_from_cpg_state_jit(
    r: torch.Tensor,
    phi: torch.Tensor,
    heading: torch.Tensor,
    s: torch.Tensor,
    h: torch.Tensor,
    d: torch.Tensor,
    g_c: torch.Tensor,
    g_p: torch.Tensor,
    L1_COS_TERM: torch.Tensor,
    L1_SIN_TERM: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute foot position from CPG state using JIT"""

    x: torch.Tensor = (
        -d * (r - 1) * torch.cos(phi) * torch.cos(heading) + s * L1_COS_TERM
    )
    y: torch.Tensor = (
        -d * (r - 1) * torch.cos(phi) * torch.sin(heading) + s * L1_SIN_TERM
    )
    sin_term = torch.sin(phi)
    z: torch.Tensor = -h + torch.where(sin_term > 0, g_c, g_p) * sin_term

    return {"x": x, "y": y, "z": z}


@torch.jit.script
def solve_joint_position_jit(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    SHIFT_L1: float,
    SHIFT_L2: float,
    SHIFT_L3: float,
    ROTATION_L1: torch.Tensor,
    ROTATION_L3: float,
) -> dict[str, torch.Tensor]:
    """Compute joint position"""

    r = (x**2 + y**2) ** 0.5 - SHIFT_L1
    c = (r**2 + z**2) ** 0.5

    alpha = torch.atan2(z, r)
    beta = torch.acos(
        torch.clamp(
            (SHIFT_L2**2 + c**2 - SHIFT_L3**2) / (2 * SHIFT_L2 * c),
            -1,
            1,
        )
    )
    gamma = torch.acos(
        torch.clamp(
            (SHIFT_L2**2 + SHIFT_L3**2 - c**2) / (2 * SHIFT_L2 * SHIFT_L3),
            -1,
            1,
        )
    )

    q1 = torch.atan2(y, x) - ROTATION_L1
    q2 = -alpha - beta
    q3 = gamma - ROTATION_L3 - math.pi

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
    }
