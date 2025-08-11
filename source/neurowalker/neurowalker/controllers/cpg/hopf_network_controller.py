import math

import torch

from .hopf_network_controller_cfg import HopfNetworkControllerCfg
from .utils import calc_psi, calc_m


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
        self.net_size: int = self._default_alpha.shape[-1]

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

        # Amplitude modulation parameter bounds
        self._mu_min = self.cfg.mu_min
        self._mu_max = self.cfg.mu_max

        # Frequency modulation parameter lower limit
        self._w_min = self.cfg.mu_min

        self.reset()

    def reset(self) -> None:
        # Setup convergence factor for each env
        self._a = torch.normal(
            self.cfg.a, self.cfg.a_std, (self.num_envs, 1), device=self.device
        )
        self._r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._theta = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._alpha = (
            torch.randn(self.num_envs, self.net_size, device=self.device)
            * self.cfg.default_alpha_std
            + self._default_alpha
        )
        self._d_r = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_v = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_theta = torch.zeros(self.num_envs, self.net_size, device=self.device)
        self._d_alpha = torch.zeros(self.num_envs, self.net_size, device=self.device)

    def _calc_mu(self, mu: torch.Tensor) -> torch.Tensor:
        return self._mu_min + (mu + 1) / 2 * (self._mu_max - self._mu_min)

    def _calc_w(self, w: torch.Tensor, w_max: torch.Tensor) -> torch.Tensor:
        return self._w_min + (w + 1) / 2 * (w_max - self._w_min)

    def _calc_coupling_term(self, r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
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
        mu: torch.Tensor,
        w: torch.Tensor,
        w_max: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        d_r = v
        d_v = self._a**2 / 4 * (self._calc_mu(mu) - r) - self._a * v
        d_theta = self._calc_w(w, w_max)
        d_alpha = w_max / 2 * self._calc_coupling_term(r, alpha)

        return d_r, d_v, d_theta, d_alpha

    def _integrate_euler(
        self,
        d_r: torch.Tensor,
        d_v: torch.Tensor,
        d_theta: torch.Tensor,
        d_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (
            self._r + (self._d_r + d_r) * self.cfg.dt / 2,
            self._v + (self._d_v + d_v) * self.cfg.dt / 2,
            torch.remainder(
                self._theta + (self._d_theta + d_theta) * self.cfg.dt / 2, 2 * math.pi
            ),
            torch.remainder(
                self._alpha + (self._d_alpha + d_alpha) * self.cfg.dt / 2, 2 * math.pi
            ),
        )

    def _integrate_rk4(
        self, mu: torch.Tensor, w: torch.Tensor, w_max: torch.Tensor
    ) -> tuple[torch.Tensor]:
        d_r1, d_v1, d_theta1, d_alpha1 = self._calc_delta_state(
            self._r, self._v, self._alpha, mu, w, w_max
        )
        r2, v2, _, alpha2 = (
            self._r + d_r1 * self.cfg.dt / 2,
            self._v + d_v1 * self.cfg.dt / 2,
            self._theta + d_theta1 * self.cfg.dt / 2,
            self._alpha + d_alpha1 * self.cfg.dt / 2,
        )
        d_r2, d_v2, d_theta2, d_alpha2 = self._calc_delta_state(
            r2, v2, alpha2, mu, w, w_max
        )
        r3, v3, _, alpha3 = (
            self._r + d_r2 * self.cfg.dt / 2,
            self._v + d_v2 * self.cfg.dt / 2,
            self._theta + d_theta2 * self.cfg.dt / 2,
            self._alpha + d_alpha2 * self.cfg.dt / 2,
        )
        d_r3, d_v3, d_theta3, d_alpha3 = self._calc_delta_state(
            r3, v3, alpha3, mu, w, w_max
        )
        r4, v4, _, alpha4 = (
            self._r + d_r3 * self.cfg.dt,
            self._v + d_v3 * self.cfg.dt,
            self._theta + d_theta3 * self.cfg.dt,
            self._alpha + d_alpha3 * self.cfg.dt,
        )
        d_r4, d_v4, d_theta4, d_alpha4 = self._calc_delta_state(
            r4, v4, alpha4, mu, w, w_max
        )

        return (
            self._r + (d_r1 + 2 * d_r2 + 2 * d_r3 + d_r4) * self.cfg.dt / 6,
            self._v + (d_v1 + 2 * d_v2 + 2 * d_v3 + d_v4) * self.cfg.dt / 6,
            self._theta
            + (d_theta1 + 2 * d_theta2 + 2 * d_theta3 + d_theta4) * self.cfg.dt / 6,
            self._alpha
            + (d_alpha1 + 2 * d_alpha2 + 2 * d_alpha3 + d_alpha4) * self.cfg.dt / 6,
        )

    def step(self, mu: torch.Tensor, w: torch.Tensor, w_max: torch.Tensor) -> None:
        d_r, d_v, d_theta, d_alpha = self._calc_delta_state(
            self._r, self._v, self._alpha, mu, w, w_max
        )

        if self.cfg.integration_method == "euler":
            self._r, self._v, self._theta, self._alpha = self._integrate_euler(
                d_r, d_v, d_theta, d_alpha
            )
        elif self.cfg.integration_method == "rk4":
            self._r, self._v, self._theta, self._alpha = self._integrate_rk4(
                mu, w, w_max
            )
        else:
            raise ValueError(
                f"Invalid integration method: {self.cfg.integration_method}"
            )

        self._d_r = d_r
        self._d_v = d_v
        self._d_theta = d_theta
        self._d_alpha = d_alpha

    @property
    def r(self) -> torch.Tensor:
        return self._r

    @property
    def d_r(self) -> torch.Tensor:
        return self._d_r

    @property
    def phi(self) -> torch.Tensor:
        return torch.remainder(self._theta + self._alpha, 2 * math.pi)

    @property
    def d_phi(self) -> torch.Tensor:
        return self._d_theta + self._d_alpha
