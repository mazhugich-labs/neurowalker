import math

import torch

from .morph_params_controller_cfg import MorphParamsControllerCfg


class MorphParamsController:
    """Morphology Parameters Controller object"""

    cfg: MorphParamsControllerCfg

    def __init__(
        self, cfg: MorphParamsControllerCfg, dt: float, num_envs: float, device: str
    ) -> None:
        self.cfg = cfg
        # Update rate is shared between all Low-Level sub-controllers
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.reset()

    def reset(self) -> None:
        """Reset morphology parameters"""
        # Stride and height must be non zero because these 2 specify starting joint positions
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

    @staticmethod
    def _verify_input_bounds(cmds: tuple[torch.Tensor]) -> bool:
        """Verify that morphology parameters bounds are in [0, 1]"""
        return any(cmd.min().item() < 0 or cmd.max().item() > 1 for cmd in cmds)

    def _fit_morph_params(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Fit morphology parameters into configuration specified bounds"""
        if self._verify_input_bounds((s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd)):
            raise ValueError("Morphology parameters out of bounds (must be in [0, 1])")

        return (
            s_cmd * (self.cfg.s_max - self.cfg.s_min) + self.cfg.s_min,
            h_cmd * (self.cfg.h_max - self.cfg.h_min) + self.cfg.h_min,
            d_cmd * (self.cfg.d_max - self.cfg.d_min) + self.cfg.d_min,
            g_c_cmd * (self.cfg.g_c_max - self.cfg.g_c_min) + self.cfg.g_c_min,
            g_p_cmd * (self.cfg.g_p_max - self.cfg.g_p_min) + self.cfg.g_p_min,
        )

    def _calc_delta_state_first_order_follower(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Caclulate delta state of morphology parameters using first-order critically damped low-pass/follower filter with time constant tau aware of controller's update rate dt"""
        f_s_cmd, f_h_cmd, f_d_cmd, f_g_c_cmd, f_g_p_cmd = self._fit_morph_params(
            s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
        )

        alpha = 1 - math.exp(-self.dt / self.cfg.mp_tau)

        delta_s = (f_s_cmd - self._s) / alpha
        delta_h = (f_h_cmd - self._h) / alpha
        delta_d = (f_d_cmd - self._d) / alpha
        delta_g_c = (f_g_c_cmd - self._g_c) / alpha
        delta_g_p = (f_g_p_cmd - self._g_p) / alpha

        return delta_s, delta_h, delta_d, delta_g_c, delta_g_p

    def _integrate_heun(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Calculate new morphology parameters using improved Euler's method of numerical integration"""
        delta_s2, delta_h2, delta_d2, delta_g_c2, delta_g_p2 = (
            self._calc_delta_state_first_order_follower(
                s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
            )
        )

        delta_s_avg = (self._delta_s + delta_s2) / 2
        delta_h_avg = (self._delta_h + delta_h2) / 2
        delta_d_avg = (self._delta_d + delta_d2) / 2
        delta_g_c_avg = (self._delta_g_c + delta_g_c2) / 2
        delta_g_p_avg = (self._delta_g_p + delta_g_p2) / 2

        return (
            self._s + delta_s_avg * self.dt,
            self._h + delta_h_avg * self.dt,
            self._d + delta_d_avg * self.dt,
            self._g_c + delta_g_c_avg * self.dt,
            self._g_p + delta_g_p_avg * self.dt,
            delta_s_avg,
            delta_h_avg,
            delta_d_avg,
            delta_g_c_avg,
            delta_g_p_avg,
        )

    def step(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate new morphology parameters"""
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

        return {
            "s": self._s,
            "h": self._h,
            "d": self._d,
            "g_c": self._g_c,
            "g_p": self._g_p,
        }

    @property
    def state(self):
        return {
            "s": self._s,
            "h": self._h,
            "d": self._d,
            "g_c": self._g_c,
            "g_p": self._g_p,
        }

    @property
    def s(self) -> torch.Tensor:
        """Leg stride [num_envs, 1]"""
        return self._s

    @property
    def h(self) -> torch.Tensor:
        """Robot height [num_envs, 1]"""
        return self._h

    @property
    def d(self) -> torch.Tensor:
        """Step length [num_envs, 1]"""
        return self._d

    @property
    def g_c(self) -> torch.Tensor:
        """Foot ground clearance [num_envs, 1]"""
        return self._g_c

    @property
    def g_p(self) -> torch.Tensor:
        """Foot ground penetration [num_envs, 1]"""
        return self._g_p
