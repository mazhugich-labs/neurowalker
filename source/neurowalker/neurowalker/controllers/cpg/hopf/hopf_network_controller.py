import math
from collections.abc import Callable

import torch

from .hopf_network_controller_cfg import HopfNetworkControllerCfg
from neurowalker.controllers.cpg.utils import calc_psi, calc_m


class HopfNetworkController:
    def __init__(
        self, cfg: HopfNetworkControllerCfg, num_envs: int, device: str
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Default alpha is the canonical phase offsets (kept FIXED)
        # shape: [1, net_size]
        self._default_alpha = torch.tensor(
            self.cfg.default_alpha, device=self.device
        ).unsqueeze(0)

        # Figure out network size (number of oscillators)
        self.net_size: int = self._default_alpha.shape[1]

        # psi is computed from default_alpha and kept fixed for robustness.
        # shape: [net_size, net_size]
        self._psi = calc_psi(self._default_alpha)

        # coupling matrix m is computed once from psi and config (kept fixed)
        self._m = calc_m(
            self._psi,
            self.cfg.coupling_cfg["self_weight"],
            self.cfg.coupling_cfg["in_group_weight"],
            self.cfg.coupling_cfg["of_group_weight"],
            self.cfg.coupling_cfg["threshold"],
            self.device,
        )

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

        self.reset()

    def reset(self) -> None:
        self._r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._theta = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._alpha = self._default_alpha.repeat(self.num_envs, 1)
        self._xi = torch.zeros(self.num_envs, 1, device=self.device)
        self._omega = torch.zeros(self.num_envs, 1, device=self.device)
        self._d_r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_theta = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_alpha = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_xi = torch.zeros(self.num_envs, 1, device=self.device)
        self._d_omega = torch.zeros(self.num_envs, 1, device=self.device)

    def _fit_mu(self, mu: torch.Tensor) -> torch.Tensor:
        """Fit amplitude modulation parameter from [-1, 1] to [self.cfg.mu_min, self.cfg.mu_max]"""
        return self.cfg.mu_min + (mu + 1) / 2 * (self.cfg.mu_max - self.cfg.mu_min)

    def _fit_w(self, w: torch.Tensor, w_max: torch.Tensor) -> torch.Tensor:
        """Fit frequency modulation parameter from [-1, 1] to [self.cfg.w_min, w_max]"""
        return self.cfg.w_min + (w + 1) / 2 * (w_max - self.cfg.w_min)

    def _fit_omega(self, omega: torch.Tensor) -> torch.Tensor:
        """Fit robot heading modulation parameter"""
        return self.cfg.omega_min + (omega + 1) / 2 * (
            self.cfg.omega_max - self.cfg.omega_min
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
        d_r = v
        d_v = self.cfg.a**2 / 4 * (self._fit_mu(mu) - r) - self.cfg.a * v
        d_theta = self._fit_w(w, w_max)
        d_alpha = w_max / 2 + self._calc_coupling_term(r, alpha)

        # Robot heading angular velocity
        d_xi = omega
        # First-order low-pass/follower filter
        d_omega = 1 / self.cfg.tau * (self._fit_omega(omega_cmd) - omega)

        return d_r, d_v, d_theta, d_alpha, d_xi, d_omega

    def _integrate_heun(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Improved Euler's numerical integration method"""
        d_r2, d_v2, d_theta2, d_alpha2, d_xi2, d_omega2 = self._calc_delta_state(
            self._r, self._v, self._alpha, self._omega, mu, w, w_max, omega_cmd
        )

        d_r_avg = (self._d_r + d_r2) / 2
        d_v_avg = (self._d_v + d_v2) / 2
        d_theta_avg = (self._d_theta + d_theta2) / 2
        d_alpha_avg = (self._d_alpha + d_alpha2) / 2
        d_xi_avg = (self._d_xi + d_xi2) / 2
        d_omega_avg = (self._d_omega + d_omega2) / 2

        return (
            self._r + d_r_avg * self.cfg.dt,
            self._v + d_v_avg * self.cfg.dt,
            torch.remainder(self._theta + d_theta_avg * self.cfg.dt, 2 * math.pi),
            torch.remainder(self._alpha + d_alpha_avg * self.cfg.dt, 2 * math.pi),
            torch.remainder(self._xi + d_xi_avg * self.cfg.dt, 2 * math.pi),
            self._omega + d_omega_avg * self.cfg.dt,
            d_r_avg,
            d_v_avg,
            d_theta_avg,
            d_alpha_avg,
            d_xi_avg,
            d_omega_avg,
        )

    def _integrate_rk4(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Runge-Kutta 4'th order numerical integration method"""
        d_r1, d_v1, d_theta1, d_alpha1, d_xi1, d_omega1 = self._calc_delta_state(
            self._r, self._v, self._alpha, self._omega, mu, w, w_max, omega_cmd
        )
        r2, v2, alpha2, omega2 = (
            self._r + d_r1 * self.cfg.dt / 2,
            self._v + d_v1 * self.cfg.dt / 2,
            self._alpha + d_alpha1 * self.cfg.dt / 2,
            self._omega + d_omega1 * self.cfg.dt / 2,
        )
        d_r2, d_v2, d_theta2, d_alpha2, d_xi2, d_omega2 = self._calc_delta_state(
            r2, v2, alpha2, omega2, mu, w, w_max, omega_cmd
        )
        r3, v3, alpha3, omega3 = (
            self._r + d_r2 * self.cfg.dt / 2,
            self._v + d_v2 * self.cfg.dt / 2,
            self._alpha + d_alpha2 * self.cfg.dt / 2,
            self._omega + d_omega2 * self.cfg.dt / 2,
        )
        d_r3, d_v3, d_theta3, d_alpha3, d_xi3, d_omega3 = self._calc_delta_state(
            r3, v3, alpha3, omega3, mu, w, w_max, omega_cmd
        )
        r4, v4, alpha4, omega4 = (
            self._r + d_r3 * self.cfg.dt,
            self._v + d_v3 * self.cfg.dt,
            self._alpha + d_alpha3 * self.cfg.dt,
            self._omega + d_omega3 * self.cfg.dt,
        )
        d_r4, d_v4, d_theta4, d_alpha4, d_xi4, d_omega4 = self._calc_delta_state(
            r4, v4, alpha4, omega4, mu, w, w_max, omega_cmd
        )

        d_r_avg = (d_r1 + 2 * d_r2 + 2 * d_r3 + d_r4) / 6
        d_v_avg = (d_v1 + 2 * d_v2 + 2 * d_v3 + d_v4) / 6
        d_theta_avg = (d_theta1 + 2 * d_theta2 + 2 * d_theta3 + d_theta4) / 6
        d_alpha_avg = (d_alpha1 + 2 * d_alpha2 + 2 * d_alpha3 + d_alpha4) / 6
        d_xi_avg = (d_xi1 + 2 * d_xi2 + 2 * d_xi3 + d_xi4) / 6
        d_omega_avg = (d_omega1 + 2 * d_omega2 + 2 * d_omega3 + d_omega4) / 6

        return (
            self._r + d_r_avg * self.cfg.dt / 6,
            self._v + d_v_avg * self.cfg.dt,
            self._theta + d_theta_avg * self.cfg.dt,
            self._alpha + d_alpha_avg * self.cfg.dt,
            torch.remainder(self._xi + d_xi_avg * self.cfg.dt, 2 * math.pi),
            self._omega + d_omega_avg * self.cfg.dt,
            d_r_avg,
            d_v_avg,
            d_theta_avg,
            d_alpha_avg,
            d_xi_avg,
            d_omega_avg,
        )

    def step(
        self,
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
        omega_cmd: float | torch.Tensor,
    ) -> None:
        """Perform one controller step"""
        (
            self._r,
            self._v,
            self._theta,
            self._alpha,
            self._xi,
            self._omega,
            self._d_r,
            self._d_v,
            self._d_theta,
            self._d_alpha,
            self._d_xi,
            self._d_omega,
        ) = self._integrate(mu, w, w_max, omega_cmd)

    @property
    def r(self) -> torch.Tensor:
        return self._r

    @property
    def phi(self) -> torch.Tensor:
        return torch.remainder(self._theta + self._alpha, 2 * math.pi)

    @property
    def xi(self) -> torch.Tensor:
        return self._xi

    @property
    def d_r(self) -> torch.Tensor:
        return self._d_r

    @property
    def d_phi(self) -> torch.Tensor:
        return self._d_theta + self._d_alpha

    @property
    def d_xi(self) -> torch.Tensor:
        return self._d_xi
