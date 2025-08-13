import math

import torch

from .inverse_kinematics_controller_cfg import InverseKinematicsControllerCfg


class InverseKinematicsController:
    """Definition of a Inverse Kinematics Controller"""

    cfg: InverseKinematicsControllerCfg

    def __init__(
        self,
        cfg: InverseKinematicsControllerCfg,
        num_envs: int,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Neurowalker specific properties
        self.__l1, self.__l2, __l3_1, __l3_2 = 0.0415, 0.078, 0.0515, 0.106
        __l3_2_rot = -math.pi / 9
        __l3_2_x = __l3_1 + __l3_2 * math.cos(__l3_2_rot)
        __l3_2_y = __l3_2 * math.sin(__l3_2_rot)
        self.__l3 = (__l3_2_x**2 + __l3_2_y**2) ** 0.5
        self.__l3_rot = math.atan2(__l3_2_y, __l3_2_x)
        self.__l1_rot = torch.tensor(
            (
                math.pi / 4,
                -math.pi / 4,
                math.pi / 2,
                -math.pi / 2,
                3 * math.pi / 4,
                -3 * math.pi / 4,
            ),
            device=self.device,
        )
        self.__l1_rot_cos_term = torch.cos(self.__l1_rot)
        self.__l1_rot_sin_term = torch.sin(self.__l1_rot)

        self._reset_morph_params()

    def _reset_state_param(self) -> None:
        pass

    def _reset_morph_params(self) -> None:
        """Reset morphological parameters"""
        # Stride and height are non zeros because my idea is that those 2 specify the initial position of each leg so they must be initialized like that
        self._s = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.s_max - self.cfg.s_min)
            + self.cfg.s_min
        )
        self._delta_s = torch.full_like(self._s, 0)
        self._h = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.h_max - self.cfg.h_min)
            + self.cfg.h_min
        )
        self._delta_h = torch.full_like(self._h, 0)

        self._d = torch.zeros((self.num_envs, 1), device=self.device)
        self._delta_d = torch.full_like(self._d, 0)
        self._g_c = torch.zeros((self.num_envs, 1), device=self.device)
        self._delta_g_c = torch.full_like(self._g_c, 0)
        self._g_p = torch.zeros((self.num_envs, 1), device=self.device)
        self._delta_g_p = torch.full_like(self._g_p, 0)

    def _clip_morph_params(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (
            torch.clamp(s_cmd, self.cfg.s_min, self.cfg.s_max),
            torch.clamp(h_cmd, self.cfg.h_min, self.cfg.h_max),
            torch.clamp(d_cmd, self.cfg.d_min, self.cfg.d_max),
            torch.clamp(g_c_cmd, self.cfg.g_c_min, self.cfg.g_c_max),
            torch.clamp(g_p_cmd, self.cfg.g_p_min, self.cfg.g_p_max),
        )

    def _calc_delta_morph_params(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Calculate delta state of morphological params based on the 1'st order low-pass/folower filter"""
        f_s_cmd, f_h_cmd, f_d_cmd, f_g_c_cmd, f_g_p_cmd = self._clip_morph_params(
            s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
        )

        delta_s = (f_s_cmd - self._s) / self.cfg.mp_tau
        delta_h = (f_h_cmd - self._h) / self.cfg.mp_tau
        delta_d = (f_d_cmd - self._d) / self.cfg.mp_tau
        delta_g_c = (f_g_c_cmd - self._g_c) / self.cfg.mp_tau
        delta_g_p = (f_g_p_cmd - self._g_p) / self.cfg.mp_tau

        return delta_s, delta_h, delta_d, delta_g_c, delta_g_p

    def _integrate_heun(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Calculate new morphological parameters using improved Euler's method of numerical integration"""
        delta_s2, delta_h2, delta_d2, delta_g_c2, delta_g_p2 = (
            self._calc_delta_morph_params(s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd)
        )

        delta_s_avg = (self._delta_s + delta_s2) / 2
        delta_h_avg = (self._delta_h + delta_h2) / 2
        delta_d_avg = (self._delta_d + delta_d2) / 2
        delta_g_c_avg = (self._delta_g_c + delta_g_c2) / 2
        delta_g_p_avg = (self._delta_g_p + delta_g_p2) / 2

        return (
            self._s + delta_s_avg * self.cfg.dt,
            self._h + delta_h_avg * self.cfg.dt,
            self._d + delta_d_avg * self.cfg.dt,
            self._g_c + delta_g_c_avg * self.cfg.dt,
            self._g_p + delta_g_p_avg * self.cfg.dt,
            delta_s2,
            delta_h2,
            delta_h2,
            delta_g_c2,
            delta_g_p2,
        )

    def _generate_cartesian_pattern(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        x = self._d * r * torch.cos(phi) * torch.cos(omega)
        y = self._d * r * torch.cos(phi) * torch.sin(omega)
        sin_term = torch.sin(phi)
        z = torch.where(
            sin_term < 0,
            -self._h + self._g_p * sin_term,
            -self._h + self._g_c * sin_term,
        )

        return (
            x + self._s * self.__l1_rot_cos_term,
            y + self._s * self.__l1_rot_sin_term,
            z,
        )

    def _solve_joint_angles(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute joint angles from generated pattern"""
        r = (x**2 + y**2) ** 0.5 - self.__l1
        c = (r**2 + z**2) ** 0.5
        alpha = torch.atan2(z, r)
        beta = torch.arccos(
            torch.clamp(
                (self.__l2**2 + c**2 - self.__l3**2) / (2 * self.__l2 * c), -1, 1
            )
        )
        gamma = torch.arccos(
            torch.clamp(
                (self.__l2**2 + self.__l3**2 - c**2) / (2 * self.__l2 * self.__l3),
                -1,
                1,
            )
        )

        return (
            torch.arctan2(y, x) - self.__l1_rot,
            -alpha - beta,
            gamma - self.__l3_rot - math.pi,
        )

    def solve_position(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Joint position mapping from CPG and morphological parameters state"""
        x, y, z = self._generate_cartesian_pattern(r, phi, omega)
        q1, q2, q3 = self._solve_joint_angles(x, y, z)

        (
            self._s,
            self._h,
            self._d,
            self._g_c,
            self._g_p,
            self._delta_s,
            self._delta_h,
            self._delta_d,
            self._delta_g_c,
            self._delta_g_p,
        ) = self._integrate_heun(s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd)

        return x, y, z, q1, q2, q3

    def solve_velocity(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Joint velocity mapping from CPG and morphological parameters state"""
        raise NotImplementedError()

    @property
    def s(self) -> torch.Tensor:
        return self._s

    @property
    def h(self) -> torch.Tensor:
        return self._h

    @property
    def d(self) -> torch.Tensor:
        return self._d

    @property
    def g_c(self) -> torch.Tensor:
        return self._g_c

    @property
    def g_p(self) -> torch.Tensor:
        return self._g_p
