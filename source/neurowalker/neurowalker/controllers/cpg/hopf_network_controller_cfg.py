import math
from collections.abc import Sequence

from isaaclab.utils import configclass


@configclass
class HopfNetworkControllerCfg:
    """Configuration for Hopf network controller"""

    dt: float = 0.02
    """Controller update rate in seconds"""

    integration_method: str = "euler"
    """Numerical integration method for controller state estimation. Available options: 'euler', 'rk4'. For reference, 'rk4' works ~4x times slower but provides smoother solution with fixed update rate"""

    a: float = 5
    a_std: float = 0.0
    """Convergence factor mean and stansard deviation for the controller. Higher values make the system converge faster. Standard deviation maybe needed during policy training to spread accross environments for the diversification and easier sim-to-real transfer"""

    default_alpha: Sequence[float] = (0, math.pi, math.pi, 0, 0, math.pi)
    default_alpha_std: float = 0.0
    """Default phase (in radians) mean and stansard deviation for each oscillator. Standard deviation maybe needed during policy training to spread accross environments for the diversification and easier sim-to-real transfer"""

    mu_min: float = 1.0
    mu_max: float = 4.0
    """Amplitude modulation for amplitude modulation parameter"""

    w_min: float = 0.0
    """Lower bound for frequency modulation parameter. Upper bound is dynamically modulated"""

    coupling_cfg: dict[str, float] = {
        "self_weight": 0.0,  # Coupling weight for an oscillator with itself (self-coupling)
        "in_group_weight": 1.0,  # Coupling weight for oscillators in the same group
        "of_group_weight": 0.0,  # Coupling weight for oscillators in different groups
        "threshold": 0.0,  # Minimal phase difference (in radians) to consider oscillators as belonging to the same group
    }
    """Coupling configuration defines how strongly oscillators influence each other. Coupling matrix is calculated from the 'default_alpha'"""
