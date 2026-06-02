import os
import os.path
import sys
from pathlib import Path

sys.path.append('../../manipulator_grasp')

import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from manipulator_grasp.arm.robot import Robot, Piper
from manipulator_grasp.utils import mj


class PiperGraspEnv:
    """MuJoCo grasping environment using a Piper 6-DOF arm."""

    def __init__(self, scene_xml_path=None, visualize=True):
        self.sim_hz = 500
        self.visualize = visualize
        self.scene_xml_path = self._resolve_scene_xml_path(scene_xml_path)

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096
        self.action_dim = 0

        # Piper built-in gripper in the simplified MJCF uses two prismatic
        # finger joints: joint7 opens in +Z, joint8 opens in -Z.
        self.gripper_open = np.array([0.035, -0.035])
        self.gripper_closed = np.array([0.0, 0.0])

        self.base_body_name = None
        self.ee_body_name = None
        self.camera_name = None

    @staticmethod
    def _resolve_scene_xml_path(scene_xml_path=None) -> str:
        """Find the Piper scene XML.

        Priority:
        1. explicit ``scene_xml_path`` argument
        2. ``PIPER_SCENE_XML`` environment variable
        3. local grasping scene, which includes ``assets/piper/piper.xml``
        4. project-level ``xml/agilex_piper/scene.xml`` from your RL project
        """
        if scene_xml_path is not None:
            path = Path(scene_xml_path)
            if not path.exists():
                raise FileNotFoundError(f"找不到指定的 Piper MuJoCo XML: {path}")
            return str(path)

        env_path = os.environ.get("PIPER_SCENE_XML")
        if env_path:
            path = Path(env_path)
            if not path.exists():
                raise FileNotFoundError(f"PIPER_SCENE_XML 指向的文件不存在: {path}")
            return str(path)

        package_dir = Path(__file__).resolve().parents[1]
        repo_root = package_dir.parent
        candidates = [
            package_dir / "assets" / "scenes" / "scene.xml",
            repo_root / "xml" / "agilex_piper" / "scene.xml",
        ]
        for path in candidates:
            if path.exists():
                return str(path)

        raise FileNotFoundError("找不到 Piper MuJoCo XML。请设置 PIPER_SCENE_XML 或放到 xml/agilex_piper/scene.xml")

    @staticmethod
    def _first_existing_name(model, obj_type, names):
        for name in names:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            if obj_id >= 0:
                return name
        return None

    def reset(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        print(f"[PiperGraspEnv] 使用 MuJoCo 场景: {self.scene_xml_path}")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.action_dim = self.mj_model.nu

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.robot = Piper()

        self.base_body_name = self._first_existing_name(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            ["base_link", "dummy_link", "piper_base", "base"],
        )
        self.ee_body_name = self._first_existing_name(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            ["ee_center_body", "gripper_center_link", "link6"],
        )
        self.camera_name = self._first_existing_name(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            ["cam", "camera", "rgb_camera"],
        )

        if self.base_body_name is None:
            self.robot.set_base(np.zeros(3))
        else:
            self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, self.base_body_name).t)

        # Match the "home" keyframe in the provided Piper MJCF.
        self.robot_q = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                mj.set_joint_q(self.mj_model, self.mj_data, joint_name, self.robot_q[i])
            elif i < self.mj_data.qpos.shape[0]:
                self.mj_data.qpos[i] = self.robot_q[i]

        # Open gripper at reset if the gripper joints exist in the model.
        for joint_name, q in zip(["joint7", "joint8"], self.gripper_open):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                mj.set_joint_q(self.mj_model, self.mj_data, joint_name, q)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Automatically align the Python kinematic model's TCP with your real
        # MuJoCo body (usually ee_center_body).  This avoids hand-tuning the
        # tool transform and keeps planning consistent with the XML model.
        if self.ee_body_name is not None:
            T_fk_without_tool = self.robot.fkine(self.robot_q)
            T_mj_ee = mj.get_body_pose(self.mj_model, self.mj_data, self.ee_body_name)
            robot_tool = T_fk_without_tool.inv() * T_mj_ee
        else:
            # Piper URDF defines gripper_center_link 135.8 mm in front of gripper_base.
            robot_tool = sm.SE3.Trans(0.0, 0.0, 0.1358)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()
        if self.action_dim:
            self.mj_data.ctrl[:] = self.make_action(self.robot_q, self.gripper_open)

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        camera = self.camera_name if self.camera_name is not None else 0
        self.mj_renderer.update_scene(self.mj_data, camera)
        self.mj_depth_renderer.update_scene(self.mj_data, camera)
        self.mj_depth_renderer.enable_depth_rendering()
        if self.visualize:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        return None

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def make_action(self, joint=None, gripper=None):
        action = np.zeros(self.action_dim)
        if joint is not None:
            action[:self.robot.dof] = np.asarray(joint)[:self.robot.dof]
        if gripper is not None and self.action_dim >= self.robot.dof + 2:
            action[self.robot.dof:self.robot.dof + 2] = np.asarray(gripper)[:2]
        elif gripper is not None and self.action_dim > self.robot.dof:
            action[-1] = np.asarray(gripper).reshape(-1)[0]
        return action

    def set_arm_action(self, action, joint):
        action[:self.robot.dof] = np.asarray(joint)[:self.robot.dof]
        return action

    def set_gripper_action(self, action, openness):
        openness = float(np.clip(openness, 0.0, 1.0))
        gripper = self.gripper_closed + openness * (self.gripper_open - self.gripper_closed)
        if self.action_dim >= self.robot.dof + 2:
            action[self.robot.dof:self.robot.dof + 2] = gripper
        elif self.action_dim > self.robot.dof:
            # The correct Piper MJCF has one gripper actuator on joint7; joint8
            # follows joint7 through an equality constraint.
            action[-1] = gripper[0]
        return action

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action[:self.action_dim]
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None:
            self.mj_viewer.sync()

    def render(self):
        camera = self.camera_name if self.camera_name is not None else 0
        self.mj_renderer.update_scene(self.mj_data, camera)
        self.mj_depth_renderer.update_scene(self.mj_data, camera)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }

    def get_camera_pose(self):
        if self.camera_name is None:
            return None
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            return None
        pos = self.mj_data.cam_xpos[cam_id].copy()
        mat = self.mj_data.cam_xmat[cam_id].reshape(3, 3).copy()
        return sm.SE3.Rt(R=mat, t=pos, check=False)


if __name__ == '__main__':
    env = PiperGraspEnv()
    env.reset()
    action = env.make_action(env.robot_q, env.gripper_open)
    for _ in range(10000):
        env.step(action)
    imgs = env.render()
    env.close()

