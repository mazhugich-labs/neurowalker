import math

import torch

from isaaclab.utils import configclass


@configclass
class InverseKinematicsControllerCfg:
    """Configuration for Inverse Kinematics Controller"""

    l1: float = 0.0415
    l2: float = 0.078
    l3: float = 0.1553956692888851
    l3_1: float = 0.0515
    l3_2: float = 0.106

    l3_2_rot = -math.pi / 9

    l1_rot: torch.Tensor = torch.tensor(
        (
            math.pi / 4,
            -math.pi / 4,
            math.pi / 2,
            -math.pi / 2,
            3 * math.pi / 4,
            -3 * math.pi / 4,
        )
    ).unsqueeze(0)

    def __post_init__(self):
        l3_x = self.l3_1 + self.l3_2 * math.cos(self.l3_2_rot)
        l3_y = self.l3_2 * math.sin(self.l3_2_rot)
        self.l3_rot = math.atan2(l3_y, l3_x)
