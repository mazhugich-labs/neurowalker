import math

import torch

from neurowalker.actuators.constants import ACCURACY, VELOCITY_LIMIT


# ---------- Controller ----------
class DynamicsController:
    def __init__(self, dt: float, tau: float = 0.5):
        self.dt = dt
        self.tau = tau

    # ---------- Intrinsic methods ----------
    def __quantize_joint_position(self, joint_pos_des: torch.Tensor):
        return torch.round(joint_pos_des / ACCURACY) * ACCURACY

    def __calc_velocity(
        self,
        joint_pos_des: torch.Tensor,
        joint_pos_cur: torch.Tensor,
        joint_vel_cur: torch.Tensor,
    ):
        joint_vel_est = (joint_pos_des - joint_pos_cur) / self.dt

        alpha = math.exp(-self.dt / self.tau)
        joint_vel_des = alpha * joint_vel_cur + (1 - alpha) * joint_vel_est

        return torch.clamp(joint_vel_des, -VELOCITY_LIMIT, VELOCITY_LIMIT)

    # ---------- Public API ----------
    def quantize_joint_position(self, joint_pos: torch.Tensor, use_jit: bool = True):
        if use_jit:
            return quantize_joint_position_jit(joint_pos, ACCURACY)

        return self.__quantize_joint_position(joint_pos)

    def calc_velocity(
        self,
        joint_pos_des: torch.Tensor,
        joint_pos_cur: torch.Tensor,
        joint_vel_cur: torch.Tensor,
        use_jit: bool = True,
    ):
        if use_jit:
            return self.__calc_velocity(joint_pos_des, joint_pos_cur, joint_vel_cur)

        return calc_velocity_jit(
            joint_pos_des,
            joint_pos_cur,
            joint_vel_cur,
            self.dt,
            self.tau,
            VELOCITY_LIMIT,
        )


# ---------- PyTorch JIT scripts ----------
@torch.jit.script
def quantize_joint_position_jit(joint_pos_des: torch.Tensor, ACCURACY: float):
    return torch.round(joint_pos_des / ACCURACY) * ACCURACY


@torch.jit.script
def calc_velocity_jit(
    joint_pos_des: torch.Tensor,
    joint_pos_cur: torch.Tensor,
    joint_vel_cur: torch.Tensor,
    dt: float,
    tau: float,
    VELOCITY_LIMIT: float,
):
    joint_vel_est = (joint_pos_des - joint_pos_cur) / dt

    alpha = math.exp(-dt / tau)
    joint_vel_des = alpha * joint_vel_cur + (1 - alpha) * joint_vel_est

    return torch.clamp(joint_vel_des, -VELOCITY_LIMIT, VELOCITY_LIMIT)
