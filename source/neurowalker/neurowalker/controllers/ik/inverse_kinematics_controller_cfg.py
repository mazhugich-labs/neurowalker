from isaaclab.utils import configclass


@configclass
class InverseKinematicsControllerCfg:
    """Configuration for Inverse Kinematics Controller"""

    dt: float = 0.02
    """Controller update rate in seconds (must be equal with CPG Network controller update rate)"""

    s_min: float = 0.05
    s_max: float = 0.1
    """Foot tip offset from the local frame in meters"""

    h_min: float = 0.05
    h_max: float = 0.15
    """Robot height lower and upper bounds in meters"""

    d_min: float = 0.02
    d_max: float = 0.07
    """Step length lower and upper bounds im meters"""

    g_c_min: float = 0.02
    g_c_max: float = 0.07
    """Foot ground clearance lower and upper bounds in meters"""

    g_p_min: float = 0.005
    g_p_max: float = 0.02
    """Foot ground penetration lower and upper bounds in meters"""

    mp_tau: float = 0.25
    """Time constant of the low-pass/follower filter to gradually change morphological parameters. Low values make the system respond faster"""

    def __post_init__(self) -> None:
        if any(
            (
                self.s_min > self.s_max,
                self.s_min < 0,
                self.s_max < 0,
                self.d_min > self.d_max,
                self.d_min < 0,
                self.d_max < 0,
                self.h_min > self.h_max,
                self.h_min < 0,
                self.h_max < 0,
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
