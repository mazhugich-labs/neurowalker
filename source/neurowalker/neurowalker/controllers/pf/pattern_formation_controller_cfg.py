from isaaclab.utils import configclass


@configclass
class PatternFormationControllerCfg:
    d_step_min: float = 0.0
    d_step_max: float = 0.07
    """Step length lower and upper bounds im meters to spread across simualtions"""

    h_min: float = 0.07
    h_max: float = 0.15
    """Robot height lower and upper bounds in meters to spread across simualtions"""

    g_c_min: float = 0.02
    g_c_max: float = 0.07
    """Foot ground clearance lower and upper bounds in meters to spread across simulations"""

    g_p_min: float = 0.0
    g_p_max: float = 0.02
    """Foot ground penetration lower and upper bounds in meters to spread across simulations"""

    def __post_init__(self) -> None:
        if any(
            (
                self.d_step_min > self.d_step_max,
                self.d_step_min < 0,
                self.d_step_max < 0,
                self.h_min > self.h_max,
                self.h_min < 0,
                self.h_max < 0,
                self.g_c_min > self.g_c_max,
                self.g_c_min < 0,
                self.g_c_max < 0,
                self.g_p_min > self.g_p_max,
                self.g_p_min < 0,
                self.g_p_max < 0,
            )
        ):
            raise ValueError("Invalid configuration")
