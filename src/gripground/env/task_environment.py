from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from gripground.config import EnvConfig
from gripground.env.simulator_adapter import SimulatorAdapter, StepResult


@dataclass(slots=True)
class SceneObjects:
    cube_id: int
    target_id: int
    table_id: int
    gripper_id: int
    finger_ids: tuple[int, int]
    arm_id: int


class PickPlacePyBulletEnv(SimulatorAdapter):
    """Physics-based pick-and-place with a Cartesian parallel-jaw gripper."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.client = p.connect(p.GUI if self.config.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.scene: SceneObjects | None = None
        self._step_count = 0
        self._instruction = "move the red cube to the green target"
        self._gripper_closed = False
        self._grasp_cid: int | None = None
        self._phase = 0
        self._ee_pos = np.array([0.43, 0.0, 0.82], dtype=np.float32)
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
        self._grasp_cid = None
        self._phase = 0
        self._ee_pos = np.array([0.43, 0.0, 0.82], dtype=np.float32)
        self._build_scene()
        return self._get_observation(), {"seed": seed, "instruction": self._instruction}

    def _build_scene(self) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.55, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client,
        )

        cube_pos = [
            0.55 + np.random.uniform(-0.07, 0.07),
            np.random.uniform(-0.12, 0.12),
            0.65,
        ]
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], physicsClientId=self.client
        )
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02, 0.02, 0.02],
            rgbaColor=[0.9, 0.08, 0.06, 1],
            physicsClientId=self.client,
        )
        cube_id = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_pos,
            physicsClientId=self.client,
        )

        target_pos = [0.69 + np.random.uniform(-0.03, 0.03), np.random.uniform(-0.10, 0.10), 0.63]
        target_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self.config.target_radius, height=0.002, physicsClientId=self.client
        )
        target_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.config.target_radius,
            length=0.002,
            rgbaColor=[0.1, 0.85, 0.1, 0.55],
            physicsClientId=self.client,
        )
        target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=target_collision,
            baseVisualShapeIndex=target_visual,
            basePosition=target_pos,
            physicsClientId=self.client,
        )

        palm_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.035, 0.045, 0.018], rgbaColor=[0.22, 0.28, 0.36, 1], physicsClientId=self.client
        )
        finger_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.012, 0.012, 0.045], rgbaColor=[0.75, 0.78, 0.82, 1], physicsClientId=self.client
        )
        gripper_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=palm_visual,
            basePosition=self._ee_pos.tolist(),
            physicsClientId=self.client,
        )
        finger_ids = tuple(
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=finger_visual,
                basePosition=(self._ee_pos + np.array([offset, 0.0, -0.05])).tolist(),
                physicsClientId=self.client,
            )
            for offset in (-0.035, 0.035)
        )
        arm_visual = p.createVisualShape(
            p.GEOM_CAPSULE, radius=0.025, length=0.34, rgbaColor=[0.38, 0.42, 0.48, 1], physicsClientId=self.client
        )
        arm_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=arm_visual,
            basePosition=[0.34, 0.0, 0.68],
            physicsClientId=self.client,
        )
        self.scene = SceneObjects(
            cube_id=cube_id,
            target_id=target_id,
            table_id=table_id,
            gripper_id=gripper_id,
            finger_ids=(int(finger_ids[0]), int(finger_ids[1])),
            arm_id=arm_id,
        )
        self._update_robot_visuals()
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

    def _update_robot_visuals(self) -> None:
        assert self.scene is not None
        p.resetBasePositionAndOrientation(
            self.scene.gripper_id, self._ee_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client
        )
        finger_offset = 0.025 if self._gripper_closed else 0.035
        for finger_id, sign in zip(self.scene.finger_ids, (-1, 1)):
            pos = self._ee_pos + np.array([sign * finger_offset, 0.0, -0.05], dtype=np.float32)
            p.resetBasePositionAndOrientation(finger_id, pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        shoulder = np.array([0.34, 0.0, 0.68], dtype=np.float32)
        direction = self._ee_pos - shoulder
        distance = float(np.linalg.norm(direction))
        if distance > 1e-6:
            direction /= distance
            orn = p.getQuaternionFromEuler([0, np.arctan2(direction[0], direction[2]), 0], physicsClientId=self.client)
            p.resetBasePositionAndOrientation(
                self.scene.arm_id,
                ((shoulder + self._ee_pos) / 2).tolist(),
                orn,
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
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        self._ee_pos = np.clip(
            self._ee_pos + action[:3] * 0.03,
            np.array([0.34, -0.32, 0.66], dtype=np.float32),
            np.array([0.78, 0.32, 0.98], dtype=np.float32),
        )
        close_cmd = bool(action[3] > 0)
        if close_cmd != self._gripper_closed:
            self._gripper_closed = close_cmd
            if close_cmd:
                self._maybe_attach()
                if self._grasp_cid is not None:
                    self._phase = 1
            elif self._grasp_cid is not None:
                p.removeConstraint(self._grasp_cid, physicsClientId=self.client)
                self._grasp_cid = None
                self._phase = 2
        self._update_robot_visuals()
        for _ in range(self.config.action_repeat):
            p.stepSimulation(physicsClientId=self.client)
        if close_cmd and self._grasp_cid is None:
            self._maybe_attach()
            if self._grasp_cid is not None:
                self._phase = 1
        if self._phase == 1 and self._grasp_cid is not None:
            cube_height = p.getBasePositionAndOrientation(self.scene.cube_id, physicsClientId=self.client)[0][2]
            if cube_height >= 0.78:
                self._phase = 2

        obs = self._get_observation()
        reward, success = self._reward_success(obs)
        truncated = self._step_count >= self.config.max_steps
        failure_category = "none" if success else ("timeout" if truncated else None)
        if not success and obs["cube_pos"][2] < 0.58:
            failure_category = "dropped_object"
        return StepResult(
            observation=obs,
            reward=reward,
            done=success,
            truncated=truncated,
            info={
                "success": success,
                "step_count": self._step_count,
                "failure_category": failure_category,
            },
        )

    def _maybe_attach(self) -> None:
        if self._grasp_cid is not None or not self._gripper_closed or self.scene is None:
            return
        cube_pos = np.asarray(
            p.getBasePositionAndOrientation(self.scene.cube_id, physicsClientId=self.client)[0], dtype=np.float32
        )
        if np.linalg.norm(self._ee_pos - cube_pos) > 0.055:
            return
        relative_pos = (cube_pos - self._ee_pos).tolist()
        self._grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.scene.gripper_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.scene.cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_pos,
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client,
        )

    def _reward_success(self, obs: dict[str, Any]) -> tuple[float, bool]:
        distance = float(np.linalg.norm(obs["cube_pos"][:2] - obs["target_pos"][:2]))
        success = distance <= self.config.target_radius * 0.8 and float(obs["cube_pos"][2]) <= 0.68
        return (1.0 if success else -distance), success

    def _get_observation(self) -> dict[str, Any]:
        assert self.scene is not None
        view = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 1.15],
            cameraTargetPosition=[0.55, 0.0, 0.64],
            cameraUpVector=[0, 0, 1],
        )
        projection = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.config.image_width / self.config.image_height,
            nearVal=0.1,
            farVal=2.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            self.config.image_width,
            self.config.image_height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (self.config.image_height, self.config.image_width, 4))[..., :3]
        cube_pos = np.asarray(
            p.getBasePositionAndOrientation(self.scene.cube_id, physicsClientId=self.client)[0], dtype=np.float32
        )
        target_pos = np.asarray(
            p.getBasePositionAndOrientation(self.scene.target_id, physicsClientId=self.client)[0], dtype=np.float32
        )
        state = np.concatenate(
            [
                self._ee_pos,
                cube_pos,
                target_pos,
                np.array(
                    [float(self._gripper_closed), float(self._grasp_cid is not None), self._phase / 3.0],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return {
            "rgb": rgb,
            "state": state,
            "instruction": self._instruction,
            "cube_pos": cube_pos,
            "target_pos": target_pos,
            "ee_pos": self._ee_pos.copy(),
        }
