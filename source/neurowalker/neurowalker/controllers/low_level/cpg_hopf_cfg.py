import math
from collections.abc import Sequence

from isaaclab.utils import configclass


@configclass
class HopfNetworkControllerCfg:
    """Configuration for Hopf network controller"""

    device: str = "cuda"

    dt: float = 0.02
    """Controller update rate (s)"""

    a: float = 5
    """Hopf oscillator/network convergence factor"""

    omega: float = 8 * math.pi
    """Hopf oscillator/network frequency needed ensure that modulation parameter w_max is in appropriate bounds (rad/s)"""

    default_alpha: Sequence[float] = (0, math.pi, math.pi, 0, 0, math.pi)
    """Configure default phase offsets between oscillators (rad)"""

    default_alpha_std: float = 0.0
    """Add random jitter to oscillators initial state for training (rad)"""

    mu_min: float = 1.0
    mu_max: float = 4.0
    """Amplitude modulation bounds applied to the raw amplitude modulation parameters"""

    w_min: float = 0.0
    """Frequency modulation lower limit applied to the raw frequncy modulation parameters. Upper bound is dynamic"""

    z_norm_min: float = 0.2
    z_norm_max: float = 1.0
    """Skill vector bounds applied to the raw skill vector"""

    coupling_cfg: dict[str, float] = {
        "self_weight": 0.0,  # coupling weight of oscillator and itself
        "in_group_weight": 1.0,  # coupling weight of oscillators within group
        "of_group_weight": 0.0,  # coupling weight of oscillators from different groups
        "threshold": 0.0,  # how close to zero mod 2pi counts as 'in group'
    }
    """Configure inter-oscillator coupling settings. Coupling matrix is calculated based on the 'default_alpha'"""
