import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.piper_grasp_env import PiperGraspEnv
from manipulator_grasp.utils import mj


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = './logs/log_rs/checkpoint-rs.tar'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def get_and_process_data(imgs):
    num_point = 20000

    # imgs = np.load(os.path.join(data_dir, 'imgs.npz'))
    color = imgs['img'] / 255.0
    depth = imgs['depth']

    height = 256
    width = 256
    fovy = np.pi / 4
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01

    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    # gg.nms()
    # gg.sort_by_score()
    # gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def generate_grasps(net, imgs, visual=False):
    end_points, cloud = get_and_process_data(imgs)
    gg = get_grasps(net, end_points)
    gg = collision_detection(gg, np.array(cloud.points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    if visual:
        vis_grasps(gg, cloud)
    return gg


def run_planners(env, robot, action, planner_array, time_array):
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)

    for timei in times:
        if timei == 0.0:
            continue
        for j in range(1, len(time_cumsum)):
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:robot.dof] = joint
                env.step(action)
                break


def run_gripper(env, action, openness_start, openness_end, steps=400):
    for i in range(steps):
        openness = openness_start + (openness_end - openness_start) * (i + 1) / steps
        if hasattr(env, "set_gripper_action"):
            env.set_gripper_action(action, openness)
        env.step(action)
    if hasattr(env, "set_gripper_action"):
        env.set_gripper_action(action, openness_end)
    for _ in range(20):
        env.step(action)


def hold_pose(env, action, steps=300):
    for _ in range(steps):
        env.step(action)


def log_gripper_state(env, label):
    try:
        joint7 = float(np.asarray(mj.get_joint_q(env.mj_model, env.mj_data, "joint7")).reshape(-1)[0])
        joint8 = float(np.asarray(mj.get_joint_q(env.mj_model, env.mj_data, "joint8")).reshape(-1)[0])
        print(f"[{label}] joint7={joint7:.4f}, joint8={joint8:.4f}")
    except Exception as exc:
        print(f"[{label}] failed to read gripper joints: {exc}")


def get_arm_joint_state(env):
    joint_values = []
    for joint_name in env.joint_names[:env.robot.dof]:
        joint_values.append(float(np.asarray(mj.get_joint_q(env.mj_model, env.mj_data, joint_name)).reshape(-1)[0]))
    return np.array(joint_values, dtype=float)


def wait_until_joint_target(env, action, target_joint, tol=1e-2, max_steps=3000, log_label=None):
    target_joint = np.asarray(target_joint, dtype=float)
    for step_idx in range(max_steps):
        env.step(action)
        current_joint = get_arm_joint_state(env)
        if np.max(np.abs(current_joint - target_joint)) <= tol:
            if log_label is not None:
                print(f"[{log_label}] reached target in {step_idx + 1} steps")
            return current_joint
    current_joint = get_arm_joint_state(env)
    if log_label is not None:
        print(f"[{log_label}] timeout, current={np.round(current_joint, 4)}, target={np.round(target_joint, 4)}")
    return current_joint


def viewer_is_running(env):
    if env.mj_viewer is None:
        return True
    is_running = getattr(env.mj_viewer, "is_running", None)
    if callable(is_running):
        return bool(is_running())
    return True


def execute_grasp_cycle(env, net, home_joint, q1, motion_time_scale, gripper_steps, settle_steps, drop_target):
    for _ in range(200):
        env.step()

    imgs = env.render()
    gg = generate_grasps(net, imgs, visual=True)
    if len(gg) == 0:
        print("[grasp_cycle] no valid grasps found, skip this round")
        return False

    robot = env.robot
    T_wb = robot.base
    # Keep the camera position fixed and rotate its optical frame 180 degrees
    # around the current camera z-axis so this stays consistent with scene.xml.
    n_wc = np.array([0.0, 1.0, 0.0])
    o_wc = np.array([1.0, 0.0, 0.5])
    t_wc = np.array([1.0, 0.6, 2.0])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    # Right-multiply an axis-alignment transform so the grasp pose matches
    # the gripper frame convention expected by the Piper pipeline.
    T_align = sm.SE3(np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]), check=False)
    T_co = T_co * T_align

    # Move the gripper centre deeper along the approach direction (z-axis)
    # so the fingers wrap around the object instead of just touching its surface.
    T_wo = T_wc * T_co * sm.SE3(0.0, 0.0, 0.0)

    # Read the actual physical joint state before we overwrite the kinematic
    # model for the z-flip check, so planner0 correctly plans from where the
    # arm really is (home_joint) to the photo pose q1.
    q0 = get_arm_joint_state(env)

    # If the grasp z-axis is nearly opposite to q1's z-axis (within ~90 deg),
    # the two-finger symmetry lets us flip 180 around z — this avoids the
    # planner taking a 180 rotation when it could just stay on the nearer side.
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    # Compare y-axis (finger direction): if opposite, flip 180 around z
    y_grasp = np.array(T_wo.R[:, 1]).flatten()
    y_t1 = np.array(T1.R[:, 1]).flatten()
    if np.dot(y_grasp, y_t1) < 0:
        T_wo = T_wo * sm.SE3(sm.SO3.Rz(np.pi))

    # planner0: home_joint -> q1
    time0 = 2 * motion_time_scale
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time0)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner0 = TrajectoryPlanner(trajectory_parameter0)

    # Robot is already at q1 — jump straight to cartesian approach
    time1 = 2 * motion_time_scale
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(0.0, 0.0, -0.1)
    position_parameter1 = LinePositionParameter(T1.t, T2.t)
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time1)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner1 = TrajectoryPlanner(trajectory_parameter1)

    time2 = 2 * motion_time_scale
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time2)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner2 = TrajectoryPlanner(trajectory_parameter2)

    action = np.zeros(env.action_dim)
    if hasattr(env, "set_gripper_action"):
        env.set_gripper_action(action, 1.0)
    run_planners(env, robot, action, [planner0, planner1, planner2], [0.0, time0, time1, time2])
    run_gripper(env, action, 1.0, 0.0, steps=gripper_steps)
    log_gripper_state(env, "after_close")

    time3 = 2 * motion_time_scale
    T4 = sm.SE3.Trans(0.0, 0.0, 0.1) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time3)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner3 = TrajectoryPlanner(trajectory_parameter3)

    time4 = 2 * motion_time_scale
    T5 = sm.SE3.Trans(drop_target[0], drop_target[1], T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time4)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner4 = TrajectoryPlanner(trajectory_parameter4)

    run_planners(env, robot, action, [planner3, planner4], [0.0, time3, time4])
    run_gripper(env, action, 0.0, 1.0, steps=gripper_steps)
    log_gripper_state(env, "after_open")

    time7 = 2 * motion_time_scale
    q_release = robot.get_joint()

    # Go straight back to home
    parameter7 = JointParameter(q_release, home_joint)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)

    run_planners(env, robot, action, [planner7], [0.0, time7])
    env.set_arm_action(action, home_joint)
    if hasattr(env, "set_gripper_action"):
        env.set_gripper_action(action, 1.0)
    reached_joint = wait_until_joint_target(env, action, home_joint, tol=1e-2, max_steps=2500, log_label="return_home")
    robot.set_joint(reached_joint)
    return True


if __name__ == '__main__':
    motion_time_scale = 0.6
    gripper_steps = 150
    settle_steps = 80

    net = get_net()

    drop_targets = [
        np.array([0.4, 0.2, 0.9]),
        np.array([0.4, 0.3, 0.9]),
        np.array([0.3, 0.2, 0.9]),
    ]

    env = PiperGraspEnv()
    env.reset()
    home_joint = get_arm_joint_state(env)
    q1 = np.array([0.0, 1.2, -1.0, 0.0, 0.8, 0.0])
    env.step()  # prime
    robot = env.robot
    action = np.zeros(env.action_dim)
    if hasattr(env, "set_gripper_action"):
        env.set_gripper_action(action, 1.0)
    robot.set_joint(home_joint)
    cycle_idx = 0

    try:
        while viewer_is_running(env):
            cycle_idx += 1
            drop_target = drop_targets[(cycle_idx - 1) % len(drop_targets)]
            print(f"[grasp_cycle] start round {cycle_idx}, drop_target=({drop_target[0]:.1f}, {drop_target[1]:.1f})")
            finished = execute_grasp_cycle(
                env,
                net,
                home_joint=home_joint,
                q1=q1,
                motion_time_scale=motion_time_scale,
                gripper_steps=gripper_steps,
                settle_steps=settle_steps,
                drop_target=drop_target,
            )
            if finished:
                print(f"[grasp_cycle] round {cycle_idx} finished, robot returned home and will capture next frame")
            else:
                print(f"[grasp_cycle] round {cycle_idx} skipped, retry capture")

        # Keep rendering after all cycles
        while viewer_is_running(env):
            env.step()
    except KeyboardInterrupt:
        print("[main] interrupted by user")
    finally:
        env.close()
