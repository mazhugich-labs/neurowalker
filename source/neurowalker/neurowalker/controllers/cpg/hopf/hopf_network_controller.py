import math
from collections.abc import Callable

import torch

from .hopf_network_controller_cfg import HopfNetworkControllerCfg


class HopfNetworkController:
    """Definition of a Hopf CPG Network Controller"""

    cfg: HopfNetworkControllerCfg

    def __init__(
        self, cfg: HopfNetworkControllerCfg, dt: float, num_envs: int, device: str
    ) -> None:
        self.cfg = cfg
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.net_size: int = self.cfg.init_state.alpha.shape[-1]

        self._integrate: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor],
        ] = None
        if self.cfg.integration_method == "heun":
            self._integrate = self._integrate_heun
        elif self.cfg.integration_method == "rk4":
            self._integrate = self._integrate_rk4
        else:
            raise ValueError(
                f"Invalid integration method: {self.cfg.integration_method}"
            )

        self.update_init_state(self.cfg.init_state.alpha)
        self.reset()

    def update_init_state(self, alpha: torch.Tensor):
        if alpha.shape.__len__() > 1:
            raise ValueError(
                "Incorrect init state shape. Shape must be [number of gaits, network size]"
            )

        self.cfg.init_state.alpha = alpha.unsqueeze(0).to(self.device)

        self._psi = self.cfg.init_state.alpha.T - self.cfg.init_state.alpha

        self._m = torch.full_like(self._psi, self.cfg.coupling_cfg["of_group_weight"])
        self._m[
            (self._psi >= -self.cfg.coupling_cfg["threshold"])
            & (self._psi <= self.cfg.coupling_cfg["threshold"])
        ] = self.cfg.coupling_cfg["in_group_weight"]
        self._m[range(self.net_size), range(self.net_size)] = self.cfg.coupling_cfg[
            "self_weight"
        ]

    def reset(self) -> None:
        self._r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._theta = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._alpha = self.cfg.init_state.alpha.repeat(self.num_envs, 1)
        self._omega = torch.zeros(self.num_envs, 1, device=self.device)
        self._delta_r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._delta_v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._delta_theta = torch.zeros(
            self.num_envs, self.net_size, device=self.device
        )
        self._delta_alpha = torch.zeros(
            self.num_envs, self.net_size, device=self.device
        )
        self._delta_omega = torch.zeros(self.num_envs, 1, device=self.device)

    @staticmethod
    def _verify_input_bounds(cmds: tuple[torch.Tensor]):
        return any(cmd.min().item() < -1 or cmd.max().item() > 1 for cmd in cmds)

    def _fit_modulation_params(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ):
        """Fit morphology parameters into configuration specified bounds"""
        if self._verify_input_bounds((mu, w, omega_cmd)):
            raise ValueError("Modulation parameters out of bounds (must be in [-1, 1])")

        return (
            self.cfg.mu_min + (mu + 1) / 2 * (self.cfg.mu_max - self.cfg.mu_min),
            self.cfg.w_min + (w + 1) / 2 * (w_max - self.cfg.w_min),
            self.cfg.omega_cmd_min
            + (omega_cmd + 1) / 2 * (self.cfg.omega_cmd_max - self.cfg.omega_cmd_min),
        )

    def _calc_coupling_term(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Calculate coupling weights for each oscillator"""
        return (
            r.unsqueeze(1)
            * self._m
            * torch.sin(alpha.unsqueeze(1) - alpha.unsqueeze(2) - self._psi)
        ).sum(2)

    def _calc_delta_state(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute state derivatives"""
        f_mu, f_w, f_omega_cmd = self._fit_modulation_params(mu, w, w_max, omega_cmd)

        delta_r = v
        delta_v = self.cfg.a**2 / 4 * (f_mu - r) - self.cfg.a * v
        delta_theta = f_w
        delta_alpha = w_max / 2 + self._calc_coupling_term(r, alpha)

        # First-order critically damped low-pass/follower filter with time constant tau aware of controller's update rate dt
        alpha = 1 - math.exp(-self.dt / self.cfg.omega_cmd_tau)
        delta_omega = (f_omega_cmd - omega) / alpha

        return delta_r, delta_v, delta_theta, delta_alpha, delta_omega

    def _integrate_heun(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Improved Euler's numerical integration method"""
        delta_r2, delta_v2, delta_theta2, delta_alpha2, delta_omega2 = (
            self._calc_delta_state(
                self._r, self._v, self._alpha, self._omega, mu, w, w_max, omega_cmd
            )
        )

        delta_r_avg = (self._delta_r + delta_r2) / 2
        delta_v_avg = (self._delta_v + delta_v2) / 2
        delta_theta_avg = (self._delta_theta + delta_theta2) / 2
        delta_alpha_avg = (self._delta_alpha + delta_alpha2) / 2
        delta_omega_avg = (self._delta_omega + delta_omega2) / 2

        return (
            self._r + delta_r_avg * self.dt,
            self._v + delta_v_avg * self.dt,
            torch.remainder(self._theta + delta_theta_avg * self.dt, 2 * math.pi),
            torch.remainder(self._alpha + delta_alpha_avg * self.dt, 2 * math.pi),
            self._omega + delta_omega_avg * self.dt,
            delta_r_avg,
            delta_v_avg,
            delta_theta_avg,
            delta_alpha_avg,
            delta_omega_avg,
        )

    def _integrate_rk4(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Runge-Kutta 4'th order numerical integration method"""
        delta_r1, delta_v1, delta_theta1, delta_alpha1, delta_omega1 = (
            self._calc_delta_state(
                self._r, self._v, self._alpha, self._omega, mu, w, w_max, omega_cmd
            )
        )
        r2, v2, alpha2, omega2 = (
            self._r + delta_r1 * self.dt / 2,
            self._v + delta_v1 * self.dt / 2,
            self._alpha + delta_alpha1 * self.dt / 2,
            self._omega + delta_omega1 * self.dt / 2,
        )
        delta_r2, delta_v2, delta_theta2, delta_alpha2, delta_omega2 = (
            self._calc_delta_state(r2, v2, alpha2, omega2, mu, w, w_max, omega_cmd)
        )
        r3, v3, alpha3, omega3 = (
            self._r + delta_r2 * self.dt / 2,
            self._v + delta_v2 * self.dt / 2,
            self._alpha + delta_alpha2 * self.dt / 2,
            self._omega + delta_omega2 * self.dt / 2,
        )
        delta_r3, delta_v3, delta_theta3, delta_alpha3, delta_omega3 = (
            self._calc_delta_state(r3, v3, alpha3, omega3, mu, w, w_max, omega_cmd)
        )
        r4, v4, alpha4, omega4 = (
            self._r + delta_r3 * self.dt,
            self._v + delta_v3 * self.dt,
            self._alpha + delta_alpha3 * self.dt,
            self._omega + delta_omega3 * self.dt,
        )
        delta_r4, delta_v4, delta_theta4, delta_alpha4, delta_omega4 = (
            self._calc_delta_state(r4, v4, alpha4, omega4, mu, w, w_max, omega_cmd)
        )

        delta_r_avg = (delta_r1 + 2 * delta_r2 + 2 * delta_r3 + delta_r4) / 6
        delta_v_avg = (delta_v1 + 2 * delta_v2 + 2 * delta_v3 + delta_v4) / 6
        delta_theta_avg = (
            delta_theta1 + 2 * delta_theta2 + 2 * delta_theta3 + delta_theta4
        ) / 6
        delta_alpha_avg = (
            delta_alpha1 + 2 * delta_alpha2 + 2 * delta_alpha3 + delta_alpha4
        ) / 6
        delta_omega_avg = (
            delta_omega1 + 2 * delta_omega2 + 2 * delta_omega3 + delta_omega4
        ) / 6

        return (
            self._r + delta_r_avg * self.dt,
            self._v + delta_v_avg * self.dt,
            torch.remainder(self._theta + delta_theta_avg * self.dt, 2 * math.pi),
            torch.remainder(self._alpha + delta_alpha_avg * self.dt, 2 * math.pi),
            self._omega + delta_omega_avg * self.dt,
            delta_r_avg,
            delta_v_avg,
            delta_theta_avg,
            delta_alpha_avg,
            delta_omega_avg,
        )

    def step(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Perform one controller step"""
        (
            self._r,
            self._v,
            self._theta,
            self._alpha,
            self._omega,
            self._delta_r,
            self._delta_v,
            self._delta_theta,
            self._delta_alpha,
            self._delta_omega,
        ) = self._integrate(mu, w, w_max, omega_cmd)

        return {
            "r": self.r,
            "delta_r": self.delta_r,
            "phi": self.phi,
            "delta_phi": self.delta_phi,
            "omega": self.omega,
            "delta_omega": self.delta_omega,
        }

    @property
    def state(self):
        return {
            "r": self.r,
            "delta_r": self.delta_r,
            "phi": self.phi,
            "delta_phi": self.delta_phi,
            "omega": self.omega,
            "delta_omega": self.delta_omega,
        }

    @property
    def r(self) -> torch.Tensor:
        return self._r

    @property
    def phi(self) -> torch.Tensor:
        return torch.remainder(self._theta + self._alpha, 2 * math.pi)

    @property
    def omega(self) -> torch.Tensor:
        return self._omega

    @property
    def delta_r(self) -> torch.Tensor:
        return self._delta_r

    @property
    def delta_phi(self) -> torch.Tensor:
        return self._delta_theta + self._delta_alpha

    @property
    def delta_omega(self) -> torch.Tensor:
        return self._delta_omega
