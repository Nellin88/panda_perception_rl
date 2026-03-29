import os
from typing import Any, Optional, SupportsFloat

import numpy as np
from gymnasium.core import ObsType

from panda_mujoco_gym.envs.panda_env import FrankaEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "reach.xml")


class FrankaReachEnv(FrankaEnv):
    def __init__(
        self,
        reward_type,
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=True,
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.0,
            goal_x_offset=0.0,
            goal_z_range=0.2,
            **kwargs,
        )

    # Override methods that depend on object joints/sites in the base task envs.
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.arm_joint_names = self._model_names.joint_names[0:7]
        self.gripper_joint_names = self._model_names.joint_names[7:9]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

    def _get_obs(self) -> dict:
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        obs = {
            "observation": np.concatenate([ee_position, ee_velocity]).copy(),
            "achieved_goal": ee_position.copy(),
            "desired_goal": self.goal.copy(),
        }

        return obs

    def _sample_goal(self) -> np.ndarray:
        goal = self.initial_mocap_position.copy()
        xy_noise = self.np_random.uniform(-self.goal_xy_range / 2, self.goal_xy_range / 2, size=2)
        z_noise = self.np_random.uniform(-self.goal_z_range / 2, self.goal_z_range / 2)

        goal[0] += xy_noise[0]
        goal[1] += xy_noise[1]
        goal[2] = np.clip(goal[2] + z_noise, 0.05, 0.8)
        return goal

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        super()._mujoco_step(action)
