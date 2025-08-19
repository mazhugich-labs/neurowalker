import math
from dataclasses import dataclass

import torch


# ---------- Configuration ----------
@dataclass(frozen=True)
class MorphModulationBounds:
    s_min: float = 0.07
    s_max: float = 0.11
    """Leg stride range"""

    h_min: float = 0.09
    h_max: float = 0.16
    """Robot height range"""

    d_min: float = 0.04
    d_max: float = 0.08
    """Step length range"""

    g_c_min: float = 0.03
    g_c_max: float = 0.07
    """Foot ground clearance range"""

    g_p_min: float = 0.005
    g_p_max: float = 0.02
    """Foot ground penetration range"""

    def __post_init__(self):
        if any(
            (
                self.s_min > self.s_max,
                self.h_min > self.h_max,
                self.d_min > self.d_max,
                self.g_c_min > self.g_c_max,
                self.g_p_min > self.g_p_max,
            )
        ):
            raise ValueError("Minimum bound cannot be greater than maximmum bound")


@dataclass(frozen=True)
class MorphGains:
    tau: float = 0.1
    """Time convergence factor. Determines how fast morphology paramteres converge to target"""

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError("Time convergence factor must greater than 0")


@dataclass(frozen=True)
class MorphCfg:
    dt: float = 0.1
    """Controller update rate"""

    bounds: MorphModulationBounds = MorphModulationBounds()
    """Controller bounds"""

    gains: MorphGains = MorphGains()
    """Controller gains"""


# -----------------------------------


# ---------- Controller ----------
class MorphController:
    cfg: MorphCfg

    def __init__(self, cfg: MorphCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

    # ---------- Intrinsic methods ----------
    def _validate_input_bounds(
        self,
        s: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor,
        g_c: torch.Tensor,
        g_p: torch.Tensor,
    ) -> None:
        if any(
            (
                s.min() < self.cfg.bounds.s_min or s.max() > self.cfg.bounds.s_max,
                h.min() < self.cfg.bounds.h_min or h.max() > self.cfg.bounds.h_max,
                d.min() < self.cfg.bounds.d_min or d.max() > self.cfg.bounds.d_max,
                g_c.min() < self.cfg.bounds.g_c_min
                or g_c.max() > self.cfg.bounds.g_c_max,
                g_p.min() < self.cfg.bounds.g_p_min
                or g_p.max() > self.cfg.bounds.g_p_max,
            )
        ):
            raise ValueError("Invalid modulation bounds")

    def _fit_s(self, s_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.s_min + s_cmd * (
            self.cfg.bounds.s_max - self.cfg.bounds.s_min
        )

    def _fit_h(self, h_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.h_min + h_cmd * (
            self.cfg.bounds.h_max - self.cfg.bounds.h_min
        )

    def _fit_d(self, d_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.d_min + d_cmd * (
            self.cfg.bounds.d_max - self.cfg.bounds.d_min
        )

    def _fit_g_c(self, g_c_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.g_c_min + g_c_cmd * (
            self.cfg.bounds.g_c_max - self.cfg.bounds.g_c_min
        )

    def _fit_g_p(self, g_p_cmd: torch.Tensor) -> torch.Tensor:
        return self.cfg.bounds.g_p_min + g_p_cmd * (
            self.cfg.bounds.g_p_max - self.cfg.bounds.g_p_min
        )

    def _calc_delta_state_heun(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        alpha = math.exp(-self.cfg.dt / self.cfg.gains.tau)

        delta_s_desired = (self._fit_s(s_cmd) - self._s) / self.cfg.dt
        delta_h_desired = (self._fit_h(h_cmd) - self._h) / self.cfg.dt
        delta_d_desired = (self._fit_d(d_cmd) - self._d) / self.cfg.dt
        delta_g_c_desired = (self._fit_g_c(g_c_cmd) - self._g_c) / self.cfg.dt
        delta_g_p_desired = (self._fit_g_p(g_p_cmd) - self._g_p) / self.cfg.dt

        delta_s = alpha * self._delta_s_prev + (1 - alpha) * delta_s_desired
        delta_h = alpha * self._delta_h_prev + (1 - alpha) * delta_h_desired
        delta_d = alpha * self._delta_d_prev + (1 - alpha) * delta_d_desired
        delta_g_c = alpha * self._delta_g_c_prev + (1 - alpha) * delta_g_c_desired
        delta_g_p = alpha * self._delta_g_p_prev + (1 - alpha) * delta_g_p_desired

        return delta_s, delta_h, delta_d, delta_g_c, delta_g_p

    def _integrate(
        self,
        delta_s: torch.Tensor,
        delta_h: torch.Tensor,
        delta_d: torch.Tensor,
        delta_g_c: torch.Tensor,
        delta_g_p: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (
            self._s + delta_s * self.cfg.dt,
            self._h + delta_h * self.cfg.dt,
            self._d + delta_d * self.cfg.dt,
            self._g_c + delta_g_c * self.cfg.dt,
            self._g_p + delta_g_p * self.cfg.dt,
        )

    # ---------- Public API ----------
    def reset(self, s: float | torch.Tensor, h: float | torch.Tensor) -> None:
        if isinstance(s, float):
            self._s = (
                torch.tensor((s), device=self.device).unsqueeze(0).repeat(self.num_envs)
            )
        elif isinstance(s, torch.Tensor):
            if s.shape[0] != self.num_envs:
                raise ValueError(
                    f"Wrong shape of 's': {s.shape}. Expected [{self.num_envs}, 1]"
                )

            self._s = s.clone()
        else:
            raise ValueError(f"Wrong dtype of 's': {type(s)}").to(self.device)
        self._delta_s_prev = torch.zeros_like(self._s)
        """Reset leg stride"""

        if isinstance(h, float):
            self._h = (
                torch.tensor((h), device=self.device).unsqueeze(0).repeat(self.num_envs)
            )
        elif isinstance(h, torch.Tensor):
            if h.shape[0] != self.num_envs:
                raise ValueError(
                    f"Wrong shape of 'h': {h.shape}. Expected [{self.num_envs}, 1]"
                )

            self._h = h.clone().to(self.device)
        else:
            raise ValueError(f"Wrong dtype of 'h': {type(h)}")
        self._delta_h_prev = torch.zeros_like(self._h)
        """Reset robot height"""

        self._d = torch.zeros(size=(self.num_envs, 1), device=self.device)
        self._delta_d_prev = torch.zeros_like(self._d)
        """Reset step length"""

        self._g_c = torch.zeros(size=(self.num_envs, 1), device=self.device)
        self._delta_g_c_prev = torch.zeros_like(self._g_c)
        """Reset foot ground clearance"""

        self._g_p = torch.zeros(size=(self.num_envs, 1), device=self.device)
        self._delta_g_p_prev = torch.zeros_like(self._g_p)

    def forward(
        self,
        s_cmd: torch.Tensor,
        h_cmd: torch.Tensor,
        d_cmd: torch.Tensor,
        g_c_cmd: torch.Tensor,
        g_p_cmd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta_s, delta_h, delta_d, delta_g_c, delta_g_p = self._calc_delta_state_heun(
            s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
        )

        self._s, self._h, self._d, self._g_c, self._g_p = self._integrate(
            delta_s, delta_h, delta_d, delta_g_c, delta_g_p
        )

        return {
            "s": self.s,
            "h": self.h,
            "d": self.d,
            "g_c": self.g_c,
            "g_p": self.g_p,
        }

    def generate_null_command(self):
        return (
            torch.zeros(size=(self.num_envs, 1), device=self.device),
            torch.zeros(size=(self.num_envs, 1), device=self.device),
            torch.zeros(size=(self.num_envs, 1), device=self.device),
            torch.zeros(size=(self.num_envs, 1), device=self.device),
            torch.zeros(size=(self.num_envs, 1), device=self.device),
        )

    @property
    def s(self) -> torch.Tensor:
        """Leg stride"""
        return self._s

    @property
    def h(self) -> torch.Tensor:
        """Robot height"""
        return self._h

    @property
    def d(self) -> torch.Tensor:
        """Step length"""
        return self._d

    @property
    def g_c(self) -> torch.Tensor:
        """Foot ground clearance"""
        return self._g_c

    @property
    def g_p(self) -> torch.Tensor:
        """Foot ground penetration"""
        return self._g_p


# ---------- PyTorch JIT scripts ----------
# TODO
