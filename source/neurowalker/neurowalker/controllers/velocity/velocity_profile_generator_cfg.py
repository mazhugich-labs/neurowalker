import torch

from isaaclab.utils import configclass


@configclass
class VelocityProfileGeneratorCfg:
    """Velocity Profile Generator object"""

    @configclass
    class InitialStateCfg:
        q_prev: torch.Tensor

    profile_type: str = "simple"
