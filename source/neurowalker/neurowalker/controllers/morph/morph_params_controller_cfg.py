from isaaclab.utils import configclass


@configclass
class MorphParamsControllerCfg:
    """Configuration for Inverse Kinematics Controller"""

    mp_tau: float = 0.25
    """Time constant of the low-pass/follower filter to gradually change morphological parameters. Low values make the system respond faster"""

    s_min: float = 0.08
    s_max: float = 0.12
    """Foot tip offset from the local frame in meters"""

    h_min: float = 0.12
    h_max: float = 0.16
    """Robot height lower and upper bounds in meters"""

    d_min: float = 0.04
    d_max: float = 0.1
    """Step length lower and upper bounds im meters"""

    g_c_min: float = 0.04
    g_c_max: float = 0.08
    """Foot ground clearance lower and upper bounds in meters"""

    g_p_min: float = 0.005
    g_p_max: float = 0.02
    """Foot ground penetration lower and upper bounds in meters"""

    def __post_init__(self) -> None:
        if any(
            (
                self.s_min > self.s_max,
                self.s_min < 0,
                self.s_max < 0,
                self.h_min > self.h_max,
                self.h_min < 0,
                self.h_max < 0,
                self.d_min > self.d_max,
                self.d_min < 0,
                self.d_max < 0,
                self.g_c_min > self.g_c_max,
                self.g_c_min < 0,
                self.g_c_max < 0,
                self.g_p_min > self.g_p_max,
                self.g_p_min < 0,
                self.g_p_max < 0,
                self.mp_tau <= 0,
            )
        ):
            raise ValueError("Invalid configuration")
