import math
from dataclasses import dataclass

import torch


# ---------- Utilities ----------
def make_coupling_bias_matrix(gait: str, device: str):
    """Factory method to make coupling bias matrix based on the gait type"""

    if gait == "tripod":
        alpha = torch.tensor(
            (0, math.pi, math.pi, 0, 0, math.pi), device=device
        ).unsqueeze(0)
    elif gait == "tetrapod":
        alpha = torch.tensor(
            (0, 2 * math.pi / 3, 2 * math.pi / 3, 4 * math.pi / 3, 4 * math.pi / 3, 0),
            device=device,
        ).unsqueeze(0)
    elif gait == "wave":
        alpha = torch.tensor(
            (
                0,
                math.pi,
                math.pi / 3,
                4 * math.pi / 3,
                2 * math.pi / 3,
                5 * math.pi / 3,
            ),
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid gait: {gait}")

    return alpha.T - alpha


def make_coupling_weight_matrix(psi: torch.Tensor, threshold: float = 1e-6):
    """Factory method to make gait-specific coupling weight matrix"""

    m = torch.full_like(psi, 0)
    m[(psi >= -threshold) & (psi <= threshold)] = 1.0

    return m


# -------------------------------


# ---------- Configuration ----------
@dataclass(frozen=True)
class HopfNetworkModulationBounds:
    mu_min: float = 0.0
    mu_max: float = 4.0
    """Amplitude modulation range"""

    omega_min: float = 0.0
    "Frequency modulation range. Maximum bound is set externally"

    def __post_init__(self):
        if any((self.mu_min > self.mu_max,)):
            raise ValueError("Minimum bound cannot be greater than maximmum bound")


@dataclass(frozen=True)
class HopfNetworkGains:
    a: float = 32
    """Convergence factor"""

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError("Convergence factor must greater than 0")


@dataclass(frozen=True)
class HopfNetworkCfg:
    dt: float = 0.02
    """Controller update rate"""

    bounds: HopfNetworkModulationBounds = HopfNetworkModulationBounds()
    """Modulation bounds"""

    gains: HopfNetworkGains = HopfNetworkGains()
    """Controller gains"""


# -----------------------------------


# ---------- Controller ----------
class HopfNetworkController:
    cfg: HopfNetworkCfg

    def __init__(self, cfg: HopfNetworkCfg, num_osc: int, num_envs: int, device: str):
        self.cfg = cfg
        self.num_osc = num_osc
        self.num_envs = num_envs
        self.device = device

        self.reset()

    # ---------- Intrinsic methods ----------
    def _validate_input_bounds(
        self,
        mu_cmd: torch.Tensor,
        omega_cmd: torch.Tensor,
        omega_max: torch.Tensor,
    ) -> None:
        if any(
            (
                mu_cmd.min() < -1 or mu_cmd.max() > 1,
                omega_cmd.min() < -1 or omega_cmd.max() > 1,
                self.cfg.bounds.omega_min > omega_max.min(),
            )
        ):
            raise ValueError("Invalid modulation bounds")

    def _fit_mu_cmd(self, mu_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.mu_min + (mu_cmd + 1) / 2 * (
            self.cfg.bounds.mu_max - self.cfg.bounds.mu_min
        )

    def _fit_omega_cmd(
        self, omega_cmd: torch.Tensor, omega_max: torch.Tensor
    ) -> torch.Tensor:
        return self.cfg.bounds.omega_min + (omega_cmd + 1) / 2 * (
            omega_max - self.cfg.bounds.omega_min
        )

    def _calc_coupling_term(self) -> torch.Tensor:
        return (
            self._r
            * self._m
            * torch.sin(self._alpha.unsqueeze(1) - self._alpha.unsqueeze(2) - self._psi)
        ).sum(dim=1)

    def __calc_delta_state_heun(
        self,
        mu_cmd: torch.Tensor,
        omega_cmd: torch.Tensor,
        omega_max: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        delta_r = (self._v + self._delta_r_prev) / 2
        delta_v = (
            self.cfg.gains.a**2 / 4 * (self._fit_mu_cmd(mu_cmd) - self._r)
            - self.cfg.gains.a * self._v
            + self._delta_v_prev
        ) / 2
        delta_theta = (
            self._fit_omega_cmd(omega_cmd, omega_max) + self._delta_theta_prev
        ) / 2
        delta_alpha = (
            omega_max / 2 + self._calc_coupling_term() + self._delta_alpha_prev
        ) / 2

        return delta_r, delta_v, delta_theta, delta_alpha

    def _integrate(
        self,
        delta_r: torch.Tensor,
        delta_v: torch.Tensor,
        delta_theta: torch.Tensor,
        delta_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (
            self._r + delta_r * self.cfg.dt,
            self._v + delta_v * self.cfg.dt,
            torch.remainder(self._theta + delta_theta * self.cfg.dt, 2 * math.pi),
            torch.remainder(self._alpha + delta_alpha * self.cfg.dt, 2 * math.pi),
        )

    # ---------- Public API ----------
    def reset(self, gait: str = "tripod") -> None:
        self._psi = make_coupling_bias_matrix(gait, self.device)
        self._m = make_coupling_weight_matrix(self._psi)

        self._r = torch.zeros(size=(self.num_envs, self.num_osc), device=self.device)
        self._delta_r_prev = torch.zeros_like(self._r)
        self._v = torch.zeros(size=(self.num_envs, self.num_osc), device=self.device)
        self._delta_v_prev = torch.zeros_like(self._v)
        self._theta = torch.zeros(
            size=(self.num_envs, self.num_osc), device=self.device
        )
        self._delta_theta_prev = torch.zeros_like(self._theta)
        self._alpha = self._psi[:, 0].unsqueeze(0).repeat(self.num_envs, 1)
        self._delta_alpha_prev = torch.zeros_like(self._alpha)

    def forward(
        self,
        mu_cmd: torch.Tensor,
        omega_cmd: torch.Tensor,
        omega_max: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_input_bounds(mu_cmd, omega_cmd, omega_max)

        delta_r, delta_v, delta_theta, delta_alpha = self.__calc_delta_state_heun(
            mu_cmd, omega_cmd, omega_max
        )
        self._r, self._v, self._theta, self._alpha = self._integrate(
            delta_r, delta_v, delta_theta, delta_alpha
        )

        self._delta_r_prev = delta_r
        self._delta_v_prev = delta_v
        self._delta_theta_prev = delta_theta
        self._delta_alpha_prev = delta_alpha

        return {
            "r": self.r,
            "phi": self.phi,
        }

    def generate_null_command(self):
        return (
            torch.zeros(size=(self.num_envs, self.num_osc), device=self.device),
            torch.zeros(size=(self.num_envs, self.num_osc), device=self.device),
            torch.ones(size=(self.num_envs, 1), device=self.device) * 2 * math.pi,
        )

    def generate_synthetic_command(self):
        return (
            torch.rand(size=(self.num_envs, self.num_osc), device=self.device),
            torch.rand(size=(self.num_envs, self.num_osc), device=self.device),
            torch.rand(size=(self.num_envs, 1), device=self.device) * math.pi
            + 2 * math.pi,
        )

    @property
    def r(self):
        """Amplitude"""
        return self._r

    @property
    def delta_r(self):
        """Amplitude rate (velocity)"""
        return self._delta_r_prev

    @property
    def v(self):
        """Velocity"""
        return self._v

    @property
    def delta_v(self):
        """Velocity rate"""
        return self._delta_v_prev

    @property
    def theta(self):
        """Modulated intrinsic phase"""
        return self._theta

    @property
    def delta_theta(self):
        """Modulated intrinsic frequency"""
        return self._delta_theta_prev

    @property
    def alpha(self):
        """Modulated gait phase"""
        return self._alpha

    @property
    def delta_alpha(self):
        """Modulated gait frequency"""
        return self._delta_alpha_prev

    @property
    def phi(self):
        """Produced phase"""
        return torch.remainder(self._theta + self._alpha, 2 * math.pi)

    @property
    def delta_phi(self):
        """Produced frequency"""
        return self._delta_theta_prev + self._delta_alpha_prev


# ---------- PyTorch JIT scripts ----------
# TODO
