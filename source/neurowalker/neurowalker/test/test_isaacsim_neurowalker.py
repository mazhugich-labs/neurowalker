"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# append custom launch argumentss
parser.add_argument(
    "--actuator_model",
    type=str,
    choices=["implicit", "ideal_pd", "dc_motor"],
    default="implicit",
    help="Actuator model that is applied to the Articulation on its creation",
)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis

from neurowalker.robots.neurowalker import make_articulation_cfg
from neurowalker.controllers.cpg.hopf import (
    HopfNetworkModulationBounds,
    HopfNetworkGains,
    HopfNetworkCfg,
    HopfNetworkController,
)
from neurowalker.controllers.morphology import (
    MorphModulationBounds,
    MorphGains,
    MorphCfg,
    MorphController,
)
from neurowalker.controllers.mechanics import KinematicsController, DynamicsController


def generate_synthetic_heading_command(num_envs: int, device: str):
    return torch.rand(size=(num_envs, 1), device=device) * 2 * math.pi - math.pi


def define_markers() -> VisualizationMarkers:
    """Define markers"""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "heading_cmd": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                ),
            ),
            "heading_actual": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 0.0)
                ),
            ),
        },
    )

    return VisualizationMarkers(marker_cfg)


class CustomSceneCfg(InteractiveSceneCfg):
    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.spawners.materials.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.1,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            )
        ),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Robot
    NeuroWalker = make_articulation_cfg(args_cli.actuator_model).replace(
        prim_path="{ENV_REGEX_NS}/NeuroWalker"
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Markers
    markers = define_markers()

    robot: Articulation = scene["NeuroWalker"]

    cpg_controller = HopfNetworkController(
        cfg=HopfNetworkCfg(
            dt=sim_dt,
            bounds=HopfNetworkModulationBounds(
                mu_min=1.0,
                mu_max=3.0,
                omega_min=0.0,
            ),
            gains=HopfNetworkGains(
                a=32,
            ),
        ),
        num_osc=6,
        num_envs=1,
        device=args_cli.device,
    )

    morph_controller = MorphController(
        cfg=MorphCfg(
            dt=0.1,
            bounds=MorphModulationBounds(
                s_min=0.07,
                s_max=0.11,
                h_min=0.09,
                h_max=0.16,
                d_min=0.04,
                d_max=0.08,
                g_c_min=0.03,
                g_c_max=0.07,
                g_p_min=0.005,
                g_p_max=0.02,
            ),
            gains=MorphGains(
                tau=0.1,
            ),
        ),
        num_envs=1,
        device=args_cli.device,
    )
    morph_controller.reset(
        s=torch.tensor(((0.07,),), device=args_cli.device),
        h=robot.data.default_root_state[:, 2].unsqueeze(1),
    )

    ik_controller = KinematicsController(device=args_cli.device)

    dyn_controller = DynamicsController(dt=sim_dt, tau=0.5)

    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            # Reset counter
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins

            # Copy default root state to the sim
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # Set target joint states
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.set_joint_velocity_target(robot.data.default_joint_vel)

            # Clear internal buffers
            scene.reset()

            # Generate synthetic heading command
            heading_cmd = generate_synthetic_heading_command(1, args_cli.device)

            print("[INFO]: Resetting NeuroWalker state...")

        # Generate synthetic null command for CPG controller (static network state)
        mu_cmd, omega_cmd, omega_max = cpg_controller.generate_null_command()

        # Perform one CPG controller step
        cpg_state_dict = cpg_controller.forward(mu_cmd, omega_cmd, omega_max)
        # Extract new CPG controller state
        r = cpg_state_dict["r"]
        phi = cpg_state_dict["phi"]

        # Generate synthetic null command for Morphology controller (static target state)
        s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd = morph_controller.generate_null_command()

        # Perform one Morphology controller step
        morph_state_dict = morph_controller.forward(
            s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
        )
        # Extract new Morphology controller state
        s = morph_state_dict["s"]
        h = morph_state_dict["h"]
        d = morph_state_dict["d"]
        g_c = morph_state_dict["g_c"]
        g_p = morph_state_dict["g_p"]

        # Compute target foot position from CPG state
        foot_pos_dict = ik_controller.solve_foot_position_from_cpg_state(
            r,
            phi,
            heading_cmd,
            s,
            h,
            d,
            g_c,
            g_p,
        )
        # Extract target foot position for each axis
        x_targ = foot_pos_dict["x"]
        y_targ = foot_pos_dict["y"]
        z_targ = foot_pos_dict["z"]

        # Compute target joint position from target foot position
        joint_pos_dict = ik_controller.solve_joint_position(x_targ, y_targ, z_targ)
        # Extract specific joint position
        joint_pos_target = torch.concat(
            (joint_pos_dict["q1"], joint_pos_dict["q2"], joint_pos_dict["q3"]), dim=1
        )

        # Quantize joint position
        joint_pos_target_quantized = dyn_controller.quantize_joint_position(
            joint_pos_target
        )

        # Compute desired joint velocity
        joint_vel_target = dyn_controller.calc_velocity(
            joint_pos_target_quantized, robot.data.joint_pos, robot.data.joint_vel
        )

        # Set target joint position and velocity
        robot.set_joint_position_target(joint_pos_target_quantized)
        robot.set_joint_velocity_target(joint_vel_target)

        # Visualize heading markers
        heading_cmd_orientation = quat_from_angle_axis(
            angle=heading_cmd,
            axis=torch.tensor([0.0, 0.0, 1.0], device=args_cli.device),
        ).squeeze(0)
        heading_actual_orientation = quat_from_angle_axis(
            angle=torch.atan2(
                robot.data.root_com_lin_vel_w[:, 1], robot.data.root_com_lin_vel_w[:, 0]
            ),
            axis=torch.tensor([0.0, 0.0, 1.0], device=args_cli.device),
        )
        marker_orientation = torch.concat(
            [heading_cmd_orientation, heading_actual_orientation]
        )
        marker_position = (robot.data.root_com_pos_w + torch.tensor([0.0, 0.0, 0.1], device=args_cli.device)).repeat(2, 1)
        markers.visualize(marker_position, marker_orientation)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.2])

    # Design scene
    scene_cfg = CustomSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
