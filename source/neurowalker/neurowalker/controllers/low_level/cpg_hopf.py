import math

import torch

from .cpg_hopf_cfg import HopfNetworkControllerCfg


def _calc_psi(default_alpha: torch.Tensor) -> torch.Tensor:
    """Calculate inter-oscillator offset matrix"""
    return default_alpha.T - default_alpha


def _calc_m(
    psi: torch.Tensor,
    self_weight: float,
    in_group_weight: float,
    of_group_weight: float,
    threshold: float,
    device: str,
) -> torch.Tensor:
    """Calculate coupling weight matrix based on the inter-oscillator phase offset matrix"""
    num_envs, num_legs, _ = psi.shape

    # Normalize phase differences mod 2pi to [0, 2pi)
    two_pi = 2 * math.pi
    phase_mod = torch.fmod(psi, two_pi)
    # Shift negatives into [0, 2pi)
    phase_mod = torch.where(phase_mod < 0, phase_mod + two_pi, phase_mod)

    # Distance to 0 mod 2pi is min(phase_mod, 2pi - phase_mod)
    dist_to_zero = torch.minimum(phase_mod, two_pi - phase_mod)

    # Create weight tensor filled with of_group_weight by default
    weights = torch.full_like(psi, of_group_weight, device=device)

    # Self connections (diagonal)
    eye_mask = (
        torch.eye(num_legs, dtype=torch.bool, device=psi.device)
        .unsqueeze(0)
        .expand(num_envs, -1, -1)
    )
    weights[eye_mask] = self_weight

    # In-group connections: phase difference close to zero modulo 2pi, excluding diagonal
    in_group_mask = (dist_to_zero <= threshold) & (~eye_mask)
    weights[in_group_mask] = in_group_weight

    return weights


class HopfNetworkController:
    def __init__(self, cfg: HopfNetworkControllerCfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = self.cfg.device

        # Default alpha is the canonical phase offsets (kept FIXED)
        # shape: [1, num_legs]
        self._default_alpha = torch.tensor(
            self.cfg.default_alpha, device=self._device
        ).unsqueeze(0)

        # Figure out network size (number of oscillators)
        self.net_size: int = self._default_alpha.size()[-1]

        # psi is computed from default_alpha and kept fixed for robustness.
        # shape: [1, num_legs, num_legs]
        self._psi = (
            _calc_psi(self._default_alpha).unsqueeze(0).expand(self.num_envs, -1, -1)
        )

        # coupling matrix m is computed once from psi and config (kept fixed)
        self._m = _calc_m(
            self._psi,
            self.cfg.coupling_cfg["self_weight"],
            self.cfg.coupling_cfg["in_group_weight"],
            self.cfg.coupling_cfg["of_group_weight"],
            self.cfg.coupling_cfg["threshold"],
            self._device,
        )

        # Default_alpha standard deviation
        self._default_alpha_std = self.cfg.default_alpha_std

        # Amplitude modulation parameters bounds
        self._mu_min = self.cfg.mu_min
        self._mu_max = self.cfg.mu_max

        # Frequency modulation parameters bounds
        self._w_min = self.cfg.mu_min

        # Skill vector bounds
        self._z_norm_min = self.cfg.z_norm_min
        self._z_norm_max = self.cfg.z_norm_max

        self.reset()

    def reset(self):
        self._r = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._v = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._theta = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._alpha = (
            torch.randn(self.num_envs, self.net_size, device=self._device)
            * self._default_alpha_std
            + self._default_alpha
        )
        self._d_r = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._d_v = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._d_theta = torch.zeros(self.num_envs, self.net_size, device=self._device)
        self._d_alpha = torch.zeros(self.num_envs, self.net_size, device=self._device)

    def _calc_mu(self, mu: torch.Tensor):
        return self._mu_min + (mu + 1) / 2 * (self._mu_max - self._mu_min)

    def _calc_w_max(self, z_norm: torch.Tensor):
        return self._z_norm_min + z_norm * (self._z_norm_max - self._z_norm_min)

    def _calc_w(self, w: torch.Tensor, w_max: torch.Tensor):
        return self._w_min + (w + 1) / 2 * (w_max - self._w_min)

    def _calc_coupling_term(self):
        return (
            self._r.unsqueeze(1)
            * self._m
            * torch.sin(self._alpha.unsqueeze(1) - self._alpha.unsqueeze(2) - self._psi)
        ).sum(2)

    def _calc_delta_state(
        self, mu: torch.Tensor, w: torch.Tensor, z_norm: torch.Tensor
    ):
        d_r = self._v
        d_v = self.cfg.a**2 / 4 * (self._calc_mu(mu) - self._r) - self.cfg.a * self._v
        w_max = self._calc_w_max(z_norm)
        d_theta = self._calc_w(w, w_max)
        d_alpha = w_max / 2 * self._calc_coupling_term()

        return d_r, d_v, d_theta, d_alpha

    def _integrate_euler(
        self,
        d_r: torch.Tensor,
        d_v: torch.Tensor,
        d_theta: torch.Tensor,
        d_alpha: torch.Tensor,
    ):
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

    def step(self, mu: torch.Tensor, w: torch.Tensor, z_norm: torch.Tensor):
        d_r, d_v, d_theta, d_alpha = self._calc_delta_state(mu, w, z_norm)

        self._r += (self._d_r + d_r) * self.cfg.dt / 2
        self._v += (self._d_v + d_v) * self.cfg.dt / 2
        self._theta = torch.remainder(
            self._theta + (self._d_theta + d_theta) * self.cfg.dt / 2, 2 * math.pi
        )
        self._alpha = torch.remainder(
            self._alpha + (self._d_alpha + d_alpha) * self.cfg.dt / 2, 2 * math.pi
        )

        self._d_r = d_r
        self._d_v = d_v
        self._d_theta = d_theta
        self._d_alpha = d_alpha

    @property
    def r(self):
        return self._r

    @property
    def d_r(self):
        return self._d_r

    @property
    def phi(self):
        return torch.remainder(self._theta + self._alpha, 2 * math.pi)

    @property
    def d_phi(self):
        return self._d_theta + self._d_alpha
