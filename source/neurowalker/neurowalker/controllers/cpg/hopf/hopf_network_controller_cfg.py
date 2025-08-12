import math
from collections.abc import Sequence

from isaaclab.utils import configclass


@configclass
class HopfNetworkControllerCfg:
    """Configuration for Hopf network controller"""

    dt: float = 0.02
    """Controller update rate in seconds"""

    integration_method: str = "heun"
    """Numerical integration method for controller state estimation. Available options: 'heun' (improved Euler's method), 'rk4'. For reference, 'rk4' works ~4x times slower but provides smoother solution"""

    a: float = 32
    """Convergence factor. Higher values make the system converge faster. Best values with an appropriate overshoot ~5%: heun - 32; rk4 - 189"""

    default_alpha: Sequence[float] = (0, math.pi, math.pi, 0, 0, math.pi)
    """Oscillators default phase offsets in radians"""

    mu_min: float = 1.0
    mu_max: float = 4.0
    """Amplitude modulation parameter bounds"""

    w_min: float = 0.0
    """Lower bound for frequency modulation parameter. Upper bound is dynamically modulated"""

    omega_min: float = -math.pi / 18
    omega_max: float = math.pi / 18
    """Robot heading modulation parameter bounds"""

    tau: float = 0.05
    """Time constans of the low-pass/follower filter for the robot heading modulation"""

    coupling_cfg: dict[str, float] = {
        "self_weight": 0.0,  # Coupling weight for an oscillator with itself (self-coupling)
        "in_group_weight": 1.0,  # Coupling weight for oscillators in the same group
        "of_group_weight": 0.0,  # Coupling weight for oscillators in different groups
        "threshold": 0.0,  # Minimal phase difference (in radians) to consider oscillators as belonging to the same group
    }
    """Coupling configuration defines how strongly oscillators influence each other. Coupling matrix is calculated from the 'default_alpha'"""

    def __post_init__(self):
        if any(
            (
                self.dt <= 0,
                self.tau <= 0,
                self.mu_min > self.mu_max,
                self.omega_min > self.omega_max,
            )
        ):
            raise ValueError("Invalid configuration")
