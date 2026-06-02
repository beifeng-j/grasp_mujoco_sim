from typing import List

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from arm.geometry import Geometry3D, Capsule
from .robot import Robot, get_transformation_mdh, wrap


class Piper(Robot):
    """Piper 6-DOF lightweight manipulator model.

    The kinematic parameters come from the Piper SDK FK implementation in
    ``graspnet-baseline/piper_sdk/piper_sdk/kinematics/piper_fk.py``.  The SDK
    uses modified-DH transforms with millimetre units; this class uses metres.
    """

    def __init__(self) -> None:
        super().__init__()

        self._dof = 6
        self.q0 = [0.0 for _ in range(self._dof)]

        # Modified DH parameters, SI units.
        self.a_array = [0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0]
        self.alpha_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2]
        self.theta_array = [
            0.0,
            -np.deg2rad(172.22),
            -np.deg2rad(102.78),
            0.0,
            0.0,
            0.0,
        ]
        self.d_array = [0.123, 0.0, 0.0, 0.25075, 0.0, 0.091]
        self.sigma_array = [0, 0, 0, 0, 0, 0]

        # Joint limits from the Piper URDF found in graspnet-baseline/piper_uou.
        self._q_lim_low = np.array([-2.618, 0.0, -2.697, -1.832, -1.22, -3.14])
        self._q_lim_up = np.array([2.618, 3.14, 0.0, 1.832, 1.22, 3.14])

        links = []
        # Approximate inertial parameters.  They are sufficient for FK/IK and
        # simple planning in this project; MuJoCo dynamics are defined in MJCF.
        for i in range(self._dof):
            links.append(
                rtb.DHLink(
                    d=self.d_array[i],
                    alpha=self.alpha_array[i],
                    a=self.a_array[i],
                    offset=self.theta_array[i],
                    mdh=True,
                    qlim=[self._q_lim_low[i], self._q_lim_up[i]],
                )
            )
        self.robot = rtb.DHRobot(links, name="piper")

        T = SE3()
        for i in range(self.dof):
            Ti = get_transformation_mdh(
                self.alpha_array[i],
                self.a_array[i],
                self.d_array[i],
                self.theta_array[i],
                self.sigma_array[i],
                0.0,
            )
            self._Ms.append(Ti.A)
            T = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            # Minimal positive-definite spatial inertia for algorithms that ask
            # Robot for dynamics-related matrices.
            mass = 0.5
            inertia = np.diag([1e-3, 1e-3, 1e-3])
            Gm = np.zeros((6, 6))
            Gm[:3, :3] = inertia
            Gm[3:, 3:] = mass * np.eye(3)
            self._Gs.append(Gm)
            self._Jms.append(0.01)

        self._Ms.append(np.eye(4))

    @property
    def q_lim_low(self) -> np.ndarray:
        return self._q_lim_low.copy()

    @property
    def q_lim_up(self) -> np.ndarray:
        return self._q_lim_up.copy()

    def ikine(self, Tep: SE3) -> np.ndarray:
        """Numerical inverse kinematics for Piper.

        Piper is not spherical-wrist compatible in the same closed-form way as
        UR5e here, so use Robotics Toolbox numerical IK with the previous joint
        state as seed.  Returns an empty array when no valid solution is found.
        """

        try:
            sol = self.robot.ikine_LM(Tep, q0=np.array(self.q0, dtype=float), ilimit=100, slimit=10, tol=1e-6)
        except TypeError:
            # Compatibility with older roboticstoolbox versions.
            sol = self.robot.ikine_LM(Tep, q0=np.array(self.q0, dtype=float))
        except Exception:
            return np.array([])

        if not getattr(sol, "success", False):
            return np.array([])

        q = np.array(sol.q, dtype=float)[: self._dof]
        q = np.clip(q, self._q_lim_low, self._q_lim_up)

        q0_s = list(map(wrap, self.q0))
        for i in range(self._dof):
            if q[i] - q0_s[i][0] > np.pi:
                q[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif q[i] - q0_s[i][0] < -np.pi:
                q[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                q[i] += q0_s[i][1] * 2 * np.pi

        return q

    def set_robot_config(self, q):
        self.q0 = np.clip(np.array(q, dtype=float), self._q_lim_low, self._q_lim_up)

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)
        if q.size != 0:
            self.q0 = q[:]

    def get_geometries(self) -> List[Geometry3D]:
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * get_transformation_mdh(
                self.alpha_array[i],
                self.a_array[i],
                self.d_array[i],
                self.theta_array[i],
                self.sigma_array[i],
                self.q0[i],
            )
            Ts.append(T)

        geometry1 = Capsule(Ts[0] * SE3.Trans(0, 0, -0.04), 0.045, 0.10)
        geometry2 = Capsule(Ts[1] * SE3.Trans(0.14, 0, 0) * SE3.Ry(np.pi / 2), 0.035, 0.28)
        geometry3 = Capsule(Ts[2] * SE3.Trans(0.12, -0.10, 0) * SE3.Ry(np.pi / 2), 0.03, 0.25)
        geometry4 = Capsule(Ts[3], 0.028, 0.12)
        geometry5 = Capsule(Ts[4], 0.025, 0.10)
        geometry6 = Capsule(Ts[5], 0.022, 0.10)
        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]


if __name__ == "__main__":
    robot = Piper()
    print(robot.fkine([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

