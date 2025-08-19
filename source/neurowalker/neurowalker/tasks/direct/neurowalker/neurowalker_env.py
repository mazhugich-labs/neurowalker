# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_angle_axis

from .neurowalker_env_cfg import NeurowalkerFlatEnvCfg, define_markers
from neurowalker.controllers.cpg.hopf import (
    HopfNetworkControllerCfg,
    HopfNetworkController,
)
from neurowalker.controllers.mechanics.kinematics import KinematicsController
from neurowalker.controllers.mechanics.dynamics import DynamicsController
from neurowalker.controllers.morphology import (
    MorphParamsControllerCfg,
    MorphParamsController,
    make_random_morph_controller_command,
)


class GenericModulator:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

    def sample_uniform_velocity_command(self, vel_lin_max: float, vel_ang_max: float):
        return torch.empty(size=(self.num_envs, 3), device=self.device).uniform_(
            -1, 1
        ) * torch.tensor(((vel_lin_max, vel_lin_max, vel_ang_max),), device=self.device)

    def sample_uniform_morph_modulation_command(self):
        return torch.rand(size=(self.num_envs, 5), device=self.device)


class NeurowalkerEnv(DirectRLEnv):
    cfg: NeurowalkerFlatEnvCfg

    def __init__(
        self, cfg: NeurowalkerFlatEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Generic modulation
        self._generic_modulator = GenericModulator(self.num_envs, self.device)

        # CPG controller
        self._cpg_controller = HopfNetworkController(
            cfg=HopfNetworkControllerCfg(
                mu_min=1.0,
                mu_max=3.0,
            ),
            init_alpha=torch.tensor(((0, math.pi, math.pi, 0, 0, math.pi),)),
            dt=self.sim.cfg.dt * self.cfg.decimation,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Morphology controller
        self._morph_controller = MorphParamsController(
            cfg=MorphParamsControllerCfg(mp_tau=0.1),
            dt=self._cpg_controller.dt,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Kinematics controller
        self._kin_controller = KinematicsController(device=self.device)

        # Dynamics controller
        self._dyn_controller = DynamicsController(dt=self._cpg_controller.dt, tau=0.1)

        # CPG modulation commands
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                # "feet_air_time",
                # "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            "tibia.*"
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._visualization_markers = define_markers()

        self._marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).to(self.device)
        self._marker_offset[:, -1] = 0.1

    def _visualize_markers(self):
        # get marker locations and orientations
        self._marker_locations = self._robot.data.root_pos_w
        self._actual_marker_orientations = quat_from_angle_axis(
            torch.atan2(
                self._robot.data.root_lin_vel_b[:, 1],
                self._robot.data.root_lin_vel_b[:, 0],
            ),
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )
        self._command_marker_orientations = quat_from_angle_axis(
            torch.atan2(self._commands[:, 1], self._commands[:, 0]),
            torch.tensor((0.0, 0.0, 1.0), device=self.device),
        )

        # offset markers so they are above the jetbot
        loc = self._marker_locations + self._marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack(
            (self._actual_marker_orientations, self._command_marker_orientations)
        )

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self._visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.tanh(actions.clone())
        self._visualize_markers()

        # Perform CPG controller step
        _ = self._cpg_controller.step(
            self._actions[:, :6],
            self._actions[:, 6:12],
            self._w_max,
            self._actions[:, -1].unsqueeze(1),
        )

        # Perform Morphologu controller step
        _ = self._morph_controller.step(*self._random_morph_params)

        # Compute target foot positions
        foot_pos_target = self._kin_controller.solve_foot_position_from_cpg_state(
            r=self._cpg_controller.r,
            phi=self._cpg_controller.phi,
            heading=self._cpg_controller.omega,
            s=self._morph_controller.s,
            h=self._morph_controller.h,
            d=self._morph_controller.d,
            g_c=self._morph_controller.g_c,
            g_p=self._morph_controller.g_p,
        )

        # Conpute target joint positions
        joint_pos_target_dict = self._kin_controller.solve_joint_position(
            x=foot_pos_target["x"],
            y=foot_pos_target["y"],
            z=foot_pos_target["z"],
        )
        joint_pos_target = torch.concat(
            (
                joint_pos_target_dict["q1"],
                joint_pos_target_dict["q2"],
                joint_pos_target_dict["q3"],
            ),
            dim=1,
        )

        # Apply quantization to computed joint position targets
        self._joint_pos_target = self._dyn_controller.quantize_joint_position(
            joint_pos_target
        )

        # Compute target joint velocities
        self._joint_vel_target = self._dyn_controller.calc_velocity(
            self._joint_pos_target,
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_pos_target)
        self._robot.set_joint_velocity_target(self._joint_vel_target)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            (
                self._cpg_controller.r,
                self._cpg_controller.phi,
                self._cpg_controller.omega,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(
                self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]
            ),
            dim=1,
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        # feet air time
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
        #     :, self._feet_ids
        # ]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self._commands[:, :2], dim=1) > 0.1
        # )
        # undesired contacts
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # is_contact = (
        #     torch.max(
        #         torch.norm(
        #             net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1
        #         ),
        #         dim=1,
        #     )[0]
        #     > 1.0
        # )
        # contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped
            * self.cfg.lin_vel_reward_scale
            * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.cfg.yaw_rate_reward_scale
            * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error
            * self.cfg.ang_vel_reward_scale
            * self.step_dt,
            "dof_torques_l2": joint_torques
            * self.cfg.joint_torque_reward_scale
            * self.step_dt,
            "dof_acc_l2": joint_accel
            * self.cfg.joint_accel_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            # "feet_air_time": air_time
            # * self.cfg.feet_air_time_reward_scale
            # * self.step_dt,
            # "undesired_contacts": contacts
            # * self.cfg.undesired_contact_reward_scale
            # * self.step_dt,
            "flat_orientation_l2": flat_orientation
            * self.cfg.flat_orientation_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1
            )[0]
            > 1.0,
            dim=1,
        )

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -0.1, 0.1
        ) * torch.tensor(((1, 1, math.pi / 2),), device=self.device)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 2] = self._morph_controller.h[env_ids].squeeze(1)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new morphology params
        self._random_morph_params = make_random_morph_controller_command(
            self.num_envs, self.device
        )

        self._w_max = (
            torch.rand((self.num_envs, 1), device=self.device) * math.pi + 2 * math.pi
        )

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

        self._visualize_markers()
