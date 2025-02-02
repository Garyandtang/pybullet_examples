import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utilsStuff.utils import skew
from planner.SE3_planner import SE3Planner
from utilsStuff.enum_class import TrajType
import matplotlib.pyplot as plt
from adaptive_control_SE3.linear_se3_error_dynamics import LinearSE3ErrorDynamics, evaluation
from environments.numerical_simulator.single_rigid_body_simulator import SingleRigidBodySimulator
from adaptive_control_SE3.adaptive_lqr.lie_algebra_adaptive import LieAlgebraStrategy
import time
from scipy.spatial.transform import Rotation  # 直接导入 Rotation

def single_test():
    init_pos = np.array([0.4, 0, 0])
    init_quat = np.array([0, 0, 0, 1])
    init_w = np.array([0, 0, 0])
    init_v = np.array([0, 0, 0])
    init_state = np.hstack([init_pos, init_quat, init_w, init_v])
    robot = SingleRigidBodySimulator()
    I_star, m_star = robot.get_true_I_m()
    lti = LinearSE3ErrorDynamics(fixed_param=True)
    config = {'type': TrajType.CONSTANT,
              'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                        'linear_vel': lti.v,
                        'angular_vel': lti.w,
                        'dt': 0.02,
                        'nTraj': 301}}
    planner = SE3Planner(config)
    ref_SE3, ref_twist, dt = planner.get_traj()
    init_traj, _ = evaluation(False, lti, config, init_state)
    A = lti.A
    B = lti.B
    Q = lti.Q
    R = lti.R
    rng = np.random
    ourEnv = LieAlgebraStrategy(Q=Q,
                                R=R,
                                A_star=A,
                                B_star=B,
                                sigma_w=0,
                                reg=1e-5,
                                tau=500,
                                actual_error_multiplier=1,
                                rls_lam=None)
    ourEnv.reset(rng)
    ourEnv.prime(1500, lti.K0, 5, rng, lti)
    I_hat = ourEnv.I_hat
    m_hat = ourEnv.m_hat
    lti.reset(fixed_param=True, I=I_hat, m=m_hat)
    adaptive_traj, _ = evaluation(False, lti, config, init_state)

    # Function to plot coordinate frames
    def plot_frames(ax, traj, color, step=30, axis_length=0.1):
        for i in range(0, traj.shape[1], step):
            # Extract position (x, y, z)
            position = traj[:3, i]

            # Extract quaternion (qw, qx, qy, qz)
            quaternion = traj[3:7, i]

            # Convert quaternion to rotation matrix
            rotation = Rotation.from_quat(quaternion).as_matrix()

            # Plot X, Y, Z axes
            ax.quiver(position[0], position[1], position[2],
                      rotation[0, 0], rotation[1, 0], rotation[2, 0],
                      color='r', length=axis_length, normalize=True)  # X axis (red)
            ax.quiver(position[0], position[1], position[2],
                      rotation[0, 1], rotation[1, 1], rotation[2, 1],
                      color='g', length=axis_length, normalize=True)  # Y axis (green)
            ax.quiver(position[0], position[1], position[2],
                      rotation[0, 2], rotation[1, 2], rotation[2, 2],
                      color='b', length=axis_length, normalize=True)  # Z axis (blue)

    # Plot 1: ref_SE3 and init_traj
    fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
    ax1.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :],
             color='blue', linestyle='-', label='Reference Trajectory')
    ax1.plot(init_traj[0, :], init_traj[1, :], init_traj[2, :],
             color='red', linestyle='--', label='Initial Trajectory')
    plot_frames(ax1, ref_SE3, color='blue', step=30)  # Frames for ref_SE3
    plot_frames(ax1, init_traj, color='red', step=30)  # Frames for init_traj
    ax1.set_xlabel('$p[0]$', fontsize=12)
    ax1.set_ylabel('$p[1]$', fontsize=12)
    ax1.set_zlabel('$p[2]$', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_title('Reference vs Initial Trajectory')

    # Save the first plot
    fig1.savefig('data/ref_vs_init_traj.jpg', dpi=300)
    print("Saved ref_vs_init_traj.jpg")

    # Plot 2: ref_SE3 and adaptive_traj
    fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
    ax2.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :],
             color='blue', linestyle='-', label='Reference Trajectory')
    ax2.plot(adaptive_traj[0, :], adaptive_traj[1, :], adaptive_traj[2, :],
             color='red', linestyle='--', label='Adaptive Trajectory')
    plot_frames(ax2, ref_SE3, color='blue', step=30)  # Frames for ref_SE3
    plot_frames(ax2, adaptive_traj, color='green', step=30)  # Frames for adaptive_traj
    ax2.set_xlabel('$p[0]$', fontsize=12)
    ax2.set_ylabel('$p[1]$', fontsize=12)
    ax2.set_zlabel('$p[2]$', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_title('Reference vs Adaptive Trajectory')

    # Save the second plot
    fig2.savefig('data/ref_vs_adaptive_traj.jpg', dpi=300)
    print("Saved ref_vs_adaptive_traj.jpg")

    # Show the plots (optional)
    plt.show()

if __name__ == '__main__':
    single_test()