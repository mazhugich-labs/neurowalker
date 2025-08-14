# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on spawning and interacting with an articulation."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg

##
# Pre-defined configs
##
from neurowalker import ASSETS_DATA_DIR
from neurowalker.controllers.cpg.hopf import HopfNetworkControllerCfg, HopfNetworkController
from neurowalker.controllers.morph import MorphParamsControllerCfg, MorphParamsController
from neurowalker.controllers.kinematics import InverseKinematicsController
from neurowalker.controllers.utils import generate_cartesian_from_cpg_mp


NEUROWALKER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/Robots/Mazhugich-Labs/neurowalker.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "hip.*": 0.0,
            "femur.*": -math.pi / 9,
            "tibia.*": -11 * math.pi / 18
        },
        joint_vel={".*": 0.0},
        pos=(0, 0, 0.3),
    ),
    actuators={
        # "actuators": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit_sim=100.0,
        #     velocity_limit_sim=100.0,
        #     stiffness=10000.0,
        #     damping=100.0,
        # ),
        "actuators": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=1.96133,
            velocity_limit=5.8,
            stiffness=80.0,
            damping=1.0,
            armature=0.0075,
            friction=0.0,
        ),
    },
)


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0],]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Articulation
    neurowalker_cfg = NEUROWALKER_CFG.copy()
    neurowalker_cfg.prim_path = "/World/Origin.*/Robot"
    neurowalker: Articulation = Articulation(cfg=neurowalker_cfg)
    neurowalker.cfg.init_state.joint_pos

    # return the scene information
    scene_entities = {"neurowalker": neurowalker}
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["neurowalker"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    decimation = 2
    controller_dt = sim_dt * decimation
    joint_pos, joint_vel = (
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    num_envs = origins.shape[0]

    cpg_controller_cfg = HopfNetworkControllerCfg(
        mu_min=1,
        mu_max=3,
    )
    cpg_controller = HopfNetworkController(cpg_controller_cfg, dt=controller_dt, num_envs=num_envs, device=args_cli.device)

    mp_controller_cfg = MorphParamsControllerCfg()
    mp_controller = MorphParamsController(mp_controller_cfg, dt=controller_dt, num_envs=num_envs, device=args_cli.device)

    ik_controller = InverseKinematicsController()

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 2000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            mu_cmd = torch.rand((num_envs, 6), device=args_cli.device) * 2 - 1
            w_cmd = torch.rand((num_envs, 6), device=args_cli.device) * 2 - 1
            w_max_cmd = torch.rand((num_envs, 1), device=args_cli.device) * math.pi + 2 * math.pi
            omega_cmd = torch.rand((num_envs, 1), device=args_cli.device) * 2 - 1

            s_cmd = torch.zeros((num_envs, 1), device=args_cli.device)
            h_cmd = torch.zeros((num_envs, 1), device=args_cli.device)
            d_cmd = torch.ones((num_envs, 1), device=args_cli.device)
            g_c_cmd = torch.zeros((num_envs, 1), device=args_cli.device)
            g_p_cmd = torch.zeros((num_envs, 1), device=args_cli.device)
            # Add target position configuration
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

        if count % decimation:
            pos_dict = generate_cartesian_from_cpg_mp(
                cpg_state=cpg_controller.state,
                mp_state=mp_controller.state,
                l1_rot=ik_controller.cfg.l1_rot.to(args_cli.device),
            )
            q_dict = ik_controller.solve_joint_position(pos_dict["x"], pos_dict["y"], pos_dict["z"])
            target_pos = torch.concat((q_dict["q1"], q_dict["q2"], q_dict["q3"]), dim=1)

            cpg_controller.step(mu_cmd, w_cmd, w_max_cmd, omega_cmd)
            mp_controller.step(s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd)

            robot.set_joint_position_target(target_pos)
            # target_velocity = torch.full((2, 18), 0.5, device=args_cli.device)
            # robot.set_joint_velocity_target(target_velocity)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
