import math

import torch

from .inverse_kinematics_controller_cfg import InverseKinematicsControllerCfg


class InverseKinematicsController:
    """Inverse Kinematics Controller object"""

    cfg: InverseKinematicsControllerCfg

    def __init__(
        self, cfg: InverseKinematicsControllerCfg = InverseKinematicsControllerCfg()
    ):
        self.cfg = cfg

    def solve_joint_position(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute joints position"""
        r = (x**2 + y**2) ** 0.5 - self.cfg.l1
        c = (r**2 + z**2) ** 0.5

        alpha = torch.atan2(z, r)
        beta = torch.acos(
            torch.clamp(
                (self.cfg.l2**2 + c**2 - self.cfg.l3**2) / (2 * self.cfg.l2 * c), -1, 1
            )
        )
        gamma = torch.acos(
            torch.clamp(
                (self.cfg.l2**2 + self.cfg.l3**2 - c**2)
                / (2 * self.cfg.l2 * self.cfg.l3),
                -1,
                1,
            )
        )

        q1 = torch.atan2(y, x) - self.cfg.l1_rot.to(x.device)
        q2 = -alpha - beta
        q3 = gamma - self.cfg.l3_rot - math.pi

        return {"q1": q1, "q2": q2, "q3": q3}

    def solve_jacobian(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        q3: torch.Tensor,
    ):
        dx_q1 = -(
            self.cfg.l1
            + self.cfg.l2 * torch.cos(q2)
            + self.cfg.l3_1 * torch.cos(q2 - q3)
            + self.cfg.l3_2 * torch.cos(q2 - q3 - self.cfg.l3_2_rot)
        ) * torch.sin(q1)
        dx_q2 = -(
            self.cfg.l2 * torch.sin(q2)
            + self.cfg.l3_1 * torch.sin(q2 - q3)
            + self.cfg.l3_2 * torch.sin(q2 - q3 - self.cfg.l3_2_rot)
        ) * torch.cos(q1)
        dx_q3 = self.cfg.l3_1 * torch.sin(q2 - q3) + self.cfg.l3_2 * torch.sin(
            q2 - q3 - self.cfg.l3_2_rot
        ) * torch.cos(q1)

        dy_q1 = -dx_q1 * torch.tan(q1)
        dy_q2 = dx_q2 * torch.tan(q1)
        dy_q3 = dx_q3 * torch.tan(q1)

        dz_q1 = None
        dz_q2 = None
        dz_q3 = None

        return torch.Tensor(
            ((dx_q1, dx_q2, dx_q3), (dy_q1, dy_q2, dy_q3), (dz_q1, dz_q2, dz_q3))
        )
