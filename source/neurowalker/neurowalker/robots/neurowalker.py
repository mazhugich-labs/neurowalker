import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from neurowalker import ASSETS_DATA_DIR
from neurowalker.actuators import (
    NEUROWALKER_IMPLICIT_ACTUATOR_CFG,
    NEUROWALKER_IDEAL_PD_ACTUATOR_CFG,
    NEUROWALKER_DC_MOTOR_CFG,
)


__spawn_cfg = sim_utils.UsdFileCfg(
    usd_path=f"{ASSETS_DATA_DIR}/Robots/Mazhugich-Labs/neurowalker.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    ),
)


__init_state_cfg = ArticulationCfg.InitialStateCfg(
    pos=(0, 0, 0.2),
    joint_pos={
        "hip.*": 0.0,
        "femur.*": -math.pi / 9,
        "tibia.*": -(math.pi / 2 + math.pi / 9),
    },
)


def make_articulation_cfg(actuator_model: str = "implicit") -> ArticulationCfg | None:
    """Factory method that helps generate Articulation Configuration based on the actuator model

    Args:
        actuator_model (str): actuator model. Available options: 'implicit', 'ideal_pd', 'dc_motor'
    """

    if actuator_model == "implicit":
        return ArticulationCfg(
            spawn=__spawn_cfg,
            init_state=__init_state_cfg,
            actuators={
                "legs": NEUROWALKER_IMPLICIT_ACTUATOR_CFG,
            },
            soft_joint_pos_limit_factor=0.95,
        )
    elif actuator_model == "ideal_pd":
        return ArticulationCfg(
            spawn=__spawn_cfg,
            init_state=__init_state_cfg,
            actuators={
                "legs": NEUROWALKER_IDEAL_PD_ACTUATOR_CFG,
            },
            soft_joint_pos_limit_factor=0.95,
        )
    elif actuator_model == "dc_motor":
        return ArticulationCfg(
            spawn=__spawn_cfg,
            init_state=__init_state_cfg,
            actuators={
                "legs": NEUROWALKER_DC_MOTOR_CFG,
            },
            soft_joint_pos_limit_factor=0.95,
        )
    else:
        raise ValueError(f"Unknown Actuator Model: {actuator_model}")
