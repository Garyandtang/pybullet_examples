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
from adaptive_control_SE3.adaptive_lqr.ofu import OFUStrategy
from adaptive_control_SE3.adaptive_lqr.ts import TSStrategy
import time
from scipy.spatial.transform import Rotation  # 直接导入 Rotation

def trajectory_comparison():
    data_size = 2000
    exitaiton = 50
    robot = SingleRigidBodySimulator()
    I_star, m_star = robot.get_true_I_m()
    lti = LinearSE3ErrorDynamics(fixed_param=True)
    config = {'type': TrajType.CONSTANT,
              'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                        'linear_vel': lti.v,
                        'angular_vel': lti.w,
                        'dt': 0.02,
                        'nTraj': 300}}
    planner = SE3Planner(config)
    ref_SE3, ref_twist, dt = planner.get_traj()
    init_traj, _ = evaluation(False, lti, config)
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
    ourEnv.prime(data_size, lti.K0, exitaiton, rng, lti)
    I_hat = ourEnv.I_hat
    m_hat = ourEnv.m_hat
    lti.reset(fixed_param=True, I=I_hat, m=m_hat)
    adaptive_traj, _ = evaluation(False, lti, config)

    lti = LinearSE3ErrorDynamics(fixed_param=True)
    A = lti.A
    B = lti.B
    Q = lti.Q
    R = lti.R
    rng = np.random
    ofuEnv = OFUStrategy(Q=Q,
                      R=R,
                      A_star=A,
                      B_star=B,
                      sigma_w=0,
                      reg=1e-5,
                      actual_error_multiplier=1,
                      rls_lam=None)
    ofuEnv.reset(rng)
    ofuEnv.prime(data_size, lti.K0, exitaiton, rng, lti)
    K = ofuEnv._current_K
    lti.update_controller(K)
    ofu_traj, _ = evaluation(False, lti, config)

    lti = LinearSE3ErrorDynamics(fixed_param=True)
    A = lti.A
    B = lti.B
    Q = lti.Q
    R = lti.R
    rng = np.random
    tsEnv = TSStrategy(Q=Q,
                        R=R,
                        A_star=A,
                        B_star=B,
                        sigma_w=0,
                        reg=1e-5,
                        tau=500,
                        actual_error_multiplier=1,
                        rls_lam=None)
    tsEnv.reset(rng)
    tsEnv.prime(data_size, lti.K0, exitaiton, rng, lti)
    K = tsEnv._current_K
    lti.update_controller(K)
    ts_traj, _ = evaluation(False, lti, config)

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
    fig1.savefig('data/ref_vs_init_traj.png', dpi=300, bbox_inches='tight')
    print("Saved ref_vs_init_traj.png")

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

    # plot ofu_traj
    fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'})
    ax3.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :],
             color='blue', linestyle='-', label='Reference Trajectory')
    ax3.plot(ofu_traj[0, :], ofu_traj[1, :], ofu_traj[2, :],
                color='red', linestyle='--', label='OFU Trajectory')
    plot_frames(ax3, ref_SE3, color='blue', step=30)  # Frames for ref_SE3
    plot_frames(ax3, ofu_traj, color='green', step=30)  # Frames for ofu_traj
    ax3.set_xlabel('$p[0]$', fontsize=12)
    ax3.set_ylabel('$p[1]$', fontsize=12)
    ax3.set_zlabel('$p[2]$', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.set_title('Reference vs OFU Trajectory')

    # plot ts_traj
    fig4, ax4 = plt.subplots(subplot_kw={'projection': '3d'})
    ax4.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :],
             color='blue', linestyle='-', label='Reference Trajectory')
    ax4.plot(ts_traj[0, :], ts_traj[1, :], ts_traj[2, :],
                color='red', linestyle='--', label='TS Trajectory')
    plot_frames(ax4, ref_SE3, color='blue', step=30)  # Frames for ref_SE3
    plot_frames(ax4, ts_traj, color='green', step=30)  # Frames for ts_traj
    ax4.set_xlabel('$p[0]$', fontsize=12)
    ax4.set_ylabel('$p[1]$', fontsize=12)
    ax4.set_zlabel('$p[2]$', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.set_title('Reference vs TS Trajectory')

    # calculate the error
    # position error
    adaptive_position_error = np.linalg.norm(ref_SE3[:3, :] - adaptive_traj[:3, :], axis=0)
    ofu_position_error = np.linalg.norm(ref_SE3[:3, :] - ofu_traj[:3, :], axis=0)
    ts_position_error = np.linalg.norm(ref_SE3[:3, :] - ts_traj[:3, :], axis=0)

    # orientation error
    adaptive_orientation_error = np.zeros(ref_SE3.shape[1])
    ofu_orientation_error = np.zeros(ref_SE3.shape[1])
    ts_orientation_error = np.zeros(ref_SE3.shape[1])
    for i in range(ref_SE3.shape[1]):
        ref_SO3 = SO3(ref_SE3[3:7, i])
        adaptive_SO3 = SO3(adaptive_traj[3:7, i])
        ofu_SO3 = SO3(ofu_traj[3:7, i])
        ts_SO3 = SO3(ts_traj[3:7, i])

        adaptive_orientation_error[i] = np.linalg.norm(ref_SO3.between(adaptive_SO3).log().coeffs())
        ofu_orientation_error[i] = np.linalg.norm(ref_SO3.between(ofu_SO3).log().coeffs())
        ts_orientation_error[i] = np.linalg.norm(ref_SO3.between(ts_SO3).log().coeffs())

    # angular velocity error
    adaptive_angular_velocity_error = np.linalg.norm(ref_twist[0:3, :] - adaptive_traj[7:10, :], axis=0)
    ofu_angular_velocity_error = np.linalg.norm(ref_twist[0:3, :] - ofu_traj[7:10, :], axis=0)
    ts_angular_velocity_error = np.linalg.norm(ref_twist[0:3, :] - ts_traj[7:10, :], axis=0)

    # linear velocity error
    adaptive_linear_velocity_error = np.linalg.norm(ref_twist[3:, :] - adaptive_traj[10:13, :], axis=0)
    ofu_linear_velocity_error = np.linalg.norm(ref_twist[3:, :] - ofu_traj[10:13, :], axis=0)
    ts_linear_velocity_error = np.linalg.norm(ref_twist[3:, :] - ts_traj[10:13, :], axis=0)

    print("Mean Position Error (Adaptive):", np.mean(adaptive_position_error))
    print("Mean Position Error (OFU):", np.mean(ofu_position_error))
    print("Mean Position Error (TS):", np.mean(ts_position_error))

    print("Mean Orientation Error (Adaptive):", np.mean(adaptive_orientation_error))
    print("Mean Orientation Error (OFU):", np.mean(ofu_orientation_error))
    print("Mean Orientation Error (TS):", np.mean(ts_orientation_error))

    print("Mean Angular Velocity Error (Adaptive):", np.mean(adaptive_angular_velocity_error))
    print("Mean Angular Velocity Error (OFU):", np.mean(ofu_angular_velocity_error))
    print("Mean Angular Velocity Error (TS):", np.mean(ts_angular_velocity_error))

    print("Mean Linear Velocity Error (Adaptive):", np.mean(adaptive_linear_velocity_error))
    print("Mean Linear Velocity Error (OFU):", np.mean(ofu_linear_velocity_error))
    print("Mean Linear Velocity Error (TS):", np.mean(ts_linear_velocity_error))




    # Show the plots (optional)
    plt.show()

if __name__ == '__main__':
    trajectory_comparison()