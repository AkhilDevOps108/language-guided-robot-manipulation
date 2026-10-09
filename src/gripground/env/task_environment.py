from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from gripground.config import EnvConfig
from gripground.env.simulator_adapter import SimulatorAdapter, StepResult


@dataclass(slots=True)
class SceneObjects:
    robot_id: int
    cube_id: int
    target_id: int
    table_id: int


class PickPlacePyBulletEnv(SimulatorAdapter):
    """PyBullet pick-and-place environment with camera observations."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.client = p.connect(p.GUI if self.config.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.scene: SceneObjects | None = None
        self._step_count = 0
        self._instruction = "move the red cube to the green target"
        self._grasp_cid: int | None = None
        self._gripper_closed = False
        self._build_scene()

    def action_space_shape(self) -> tuple[int, ...]:
        return (4,)

    def close(self) -> None:
        p.disconnect(physicsClientId=self.client)

    def reset(self, seed: int | None = None, instruction: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._instruction = instruction or "move the red cube to the green target"
        self._step_count = 0
        self._gripper_closed = False
        if self._grasp_cid is not None:
            p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
            self._grasp_cid = None
        self._build_scene()
        obs = self._get_observation()
        info = {"seed": seed, "instruction": self._instruction}
        return obs, info

    def _build_scene(self) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.55, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        cube_pos = [0.55 + np.random.uniform(-0.1, 0.1), np.random.uniform(-0.15, 0.15), 0.65]
        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], physicsClientId=self.client)
        cube_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[1, 0.1, 0.1, 1], physicsClientId=self.client
        )
        cube_id = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=cube_pos,
            physicsClientId=self.client,
        )
        target_pos = [0.7, np.random.uniform(-0.12, 0.12), 0.63]
        target_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self.config.target_radius, height=0.001, physicsClientId=self.client
        )
        target_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.config.target_radius,
            length=0.001,
            rgbaColor=[0.1, 1, 0.1, 0.5],
            physicsClientId=self.client,
        )
        target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=target_col,
            baseVisualShapeIndex=target_vis,
            basePosition=target_pos,
            physicsClientId=self.client,
        )
        self.scene = SceneObjects(robot_id=robot_id, cube_id=cube_id, target_id=target_id, table_id=table_id)
        self._reset_robot_pose()
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

    def _reset_robot_pose(self) -> None:
        assert self.scene is not None
        neutral = [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.8]
        for jid, joint_val in enumerate(neutral):
            p.resetJointState(self.scene.robot_id, jid, joint_val, physicsClientId=self.client)
        self._set_gripper(open_gripper=True)

    def _set_gripper(self, open_gripper: bool) -> None:
        assert self.scene is not None
        target = 0.04 if open_gripper else 0.0
        for jid in (9, 10):
            p.setJointMotorControl2(
                self.scene.robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=60,
                physicsClientId=self.client,
            )

    def step(self, action: np.ndarray) -> StepResult:
        if self.scene is None:
            raise RuntimeError("Environment scene is not initialized")
        if action.shape != (4,) or not np.isfinite(action).all():
            return StepResult(
                observation=self._get_observation(),
                reward=-1.0,
                done=True,
                truncated=False,
                info={"failure_category": "invalid_action"},
            )
        self._step_count += 1
        clipped = np.clip(action.astype(np.float32), -1.0, 1.0)
        ee_state = p.getLinkState(self.scene.robot_id, 11, computeForwardKinematics=True, physicsClientId=self.client)
        ee_pos = np.array(ee_state[0], dtype=np.float32)
        target_ee = ee_pos + clipped[:3] * 0.03
        target_ee[2] = np.clip(target_ee[2], 0.61, 0.95)
        self._apply_ik(target_ee)

        close_cmd = clipped[3] > 0
        if close_cmd and not self._gripper_closed:
            self._set_gripper(open_gripper=False)
            self._gripper_closed = True
        elif (not close_cmd) and self._gripper_closed:
            self._set_gripper(open_gripper=True)
            self._gripper_closed = False
            if self._grasp_cid is not None:
                p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
                self._grasp_cid = None

        for _ in range(self.config.action_repeat):
            p.stepSimulation(physicsClientId=self.client)
        self._maybe_attach()

        obs = self._get_observation()
        reward, success = self._reward_success(obs)
        done = success
        truncated = self._step_count >= self.config.max_steps
        info: dict[str, Any] = {
            "success": success,
            "step_count": self._step_count,
            "failure_category": None,
        }
        if done:
            info["failure_category"] = "none"
        elif truncated:
            info["failure_category"] = "timeout"
        elif obs["cube_pos"][2] < 0.58:
            info["failure_category"] = "dropped_object"
        return StepResult(observation=obs, reward=reward, done=done, truncated=truncated, info=info)

    def _apply_ik(self, target_ee: np.ndarray) -> None:
        assert self.scene is not None
        orn = p.getQuaternionFromEuler([math.pi, 0, 0], physicsClientId=self.client)
        joints = p.calculateInverseKinematics(
            self.scene.robot_id,
            11,
            target_ee.tolist(),
            orn,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        for jid in range(7):
            p.setJointMotorControl2(
                self.scene.robot_id,
                jid,
                p.POSITION_CONTROL,
                targetPosition=float(joints[jid]),
                force=150,
                physicsClientId=self.client,
            )

    def _maybe_attach(self) -> None:
        if self._grasp_cid is not None or not self._gripper_closed or self.scene is None:
            return
        ee_pos = np.array(
            p.getLinkState(self.scene.robot_id, 11, computeForwardKinematics=True, physicsClientId=self.client)[0],
            dtype=np.float32,
        )
        cube_pos = np.array(p.getBasePositionAndOrientation(self.scene.cube_id, physicsClientId=self.client)[0], dtype=np.float32)
        if np.linalg.norm(ee_pos - cube_pos) < 0.05:
            self._grasp_cid = p.createConstraint(
                parentBodyUniqueId=self.scene.robot_id,
                parentLinkIndex=11,
                childBodyUniqueId=self.scene.cube_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.05],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.client,
            )

    def _reward_success(self, obs: dict[str, Any]) -> tuple[float, bool]:
        cube_xy = obs["cube_pos"][:2]
        target_xy = obs["target_pos"][:2]
        dist = float(np.linalg.norm(cube_xy - target_xy))
        success = dist <= self.config.target_radius * 0.85 and float(obs["cube_pos"][2]) <= 0.67
        reward = 1.0 if success else -dist
        return reward, success

    def _get_observation(self) -> dict[str, Any]:
        assert self.scene is not None
        view = p.computeViewMatrix(cameraEyePosition=[1.0, 0.0, 1.0], cameraTargetPosition=[0.55, 0.0, 0.62], cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=2.0)
        _, _, rgb, _, _ = p.getCameraImage(
            self.config.image_width,
            self.config.image_height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.reshape(np.array(rgb, dtype=np.uint8), (self.config.image_height, self.config.image_width, 4))[..., :3]
        cube_pos = np.array(p.getBasePositionAndOrientation(self.scene.cube_id, physicsClientId=self.client)[0], dtype=np.float32)
        target_pos = np.array(p.getBasePositionAndOrientation(self.scene.target_id, physicsClientId=self.client)[0], dtype=np.float32)
        ee_pos = np.array(
            p.getLinkState(self.scene.robot_id, 11, computeForwardKinematics=True, physicsClientId=self.client)[0],
            dtype=np.float32,
        )
        state = np.concatenate(
            [
                ee_pos,
                cube_pos,
                target_pos,
                np.array([1.0 if self._gripper_closed else 0.0, 1.0 if self._grasp_cid is not None else 0.0], dtype=np.float32),
            ]
        ).astype(np.float32)
        return {
            "rgb": rgb,
            "state": state,
            "instruction": self._instruction,
            "cube_pos": cube_pos,
            "target_pos": target_pos,
            "ee_pos": ee_pos,
        }

