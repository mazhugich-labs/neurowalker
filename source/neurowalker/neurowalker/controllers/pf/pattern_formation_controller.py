import torch

from .pattern_formation_controller_cfg import PatternFormationControllerCfg


class PatternFormationController:
    def __init__(
        self,
        cfg: PatternFormationControllerCfg,
        num_envs: int,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.reset()

    def reset(self) -> None:
        self._d_step = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.d_step_max - self.cfg.d_step_min)
            + self.cfg.d_step_min
        )
        self._h = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.h_max - self.cfg.h_min)
            + self.cfg.h_min
        )
        self._g_c = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.g_c_max - self.cfg.g_c_min)
            + self.cfg.g_c_min
        )
        self._g_p = (
            torch.rand((self.num_envs, 1), device=self.device)
            * (self.cfg.g_p_max - self.cfg.g_p_min)
            + self.cfg.g_p_min
        )

    def solve_desired_pose(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        xi: torch.Tensor,
        d_step: float = None,
        h: float = None,
        g_c: float = None,
        g_p: float = None,
    ) -> tuple[torch.Tensor]:
        X = self._d_step * (r) * torch.cos(phi) * torch.cos(xi)
        Y = self._d_step * (r) * torch.cos(phi) * torch.sin(xi)
        sin_term = torch.sin(phi)
        Z = torch.where(
            sin_term < 0,
            -self._h + self._g_p * sin_term,
            -self._h + self._g_c * sin_term,
        )

        return X, Y, Z

    @property
    def d_step(self) -> torch.Tensor:
        return self._d_step

    @property
    def h(self) -> torch.Tensor:
        return self._h

    @property
    def g_c(self) -> torch.Tensor:
        return self._g_c

    @property
    def g_p(self) -> torch.Tensor:
        return self._g_p
