from isaaclab.actuators import (
    ImplicitActuatorCfg,
    IdealPDActuatorCfg,
    DCMotorCfg,
)

from .constants import (
    SATURATION_EFFORT,
    EFFORT_LIMIT,
    VELOCITY_LIMIT,
    STIFFNESS,
    DAMPING,
    ARMATURE,
    FRICTION,
)


NEUROWALKER_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit_sim=EFFORT_LIMIT,
    velocity_limit_sim=VELOCITY_LIMIT,
    stiffness={".*": STIFFNESS},
    damping={".*": DAMPING},
    armature=ARMATURE,
    friction=FRICTION,
)
"""Neurowalker configuration for Implicit Actuator Model"""


NEUROWALKER_IDEAL_PD_ACTUATOR_CFG = IdealPDActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=EFFORT_LIMIT,
    velocity_limit=VELOCITY_LIMIT,
    stiffness={".*": STIFFNESS},
    damping={".*": DAMPING},
    armature=ARMATURE,
    friction=FRICTION,
)
"""Neurowalker configuration for Ideal PD Actuator Model"""


NEUROWALKER_DC_MOTOR_CFG = DCMotorCfg(
    joint_names_expr=[".*"],
    effort_limit=EFFORT_LIMIT,
    velocity_limit=VELOCITY_LIMIT,
    stiffness={".*": STIFFNESS},
    damping={".*": DAMPING},
    armature=ARMATURE,
    friction=FRICTION,
    saturation_effort=SATURATION_EFFORT,
)
"""Neurowalker configuration for DC Motor Model"""
