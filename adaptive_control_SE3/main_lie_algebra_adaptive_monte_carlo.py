import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utilsStuff.utils import skew
from planner.SE3_planner import SE3Planner
from utilsStuff.enum_class import TrajType
import matplotlib.pyplot as plt
from adaptive_control_SE3.linear_se3_error_dynamics import LinearSE3ErrorDynamics
from environments.numerical_simulator.single_rigid_body_simulator import SingleRigidBodySimulator
from adaptive_control_SE3.adaptive_lqr.lie_algebra_adaptive import LieAlgebraStrategy
import time
def MonteCarloSimulation():
    totalSim = 50
    step_vector = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # step_vector = np.array([0, 200, 400])
    I_error = np.zeros((totalSim, len(step_vector)))
    m_error = np.zeros((totalSim, len(step_vector)))
    times = np.zeros((totalSim, len(step_vector)))

    robot = SingleRigidBodySimulator()
    I_star, m_star = robot.get_true_I_m()
    for i in range(len(step_vector)):
        for sim in range(totalSim):
            start_time = time.time()
            data_size = step_vector[i]
            lti_config = {'fixed_param': False,
                            'I': I_star,
                            'm': m_star,
                            'v': np.array([2, 0, 0.2]),
                            'w': np.array([0, 0, 1])}
            lti = LinearSE3ErrorDynamics(lti_config)
            if data_size == 0:
                I_hat = lti.I
                m_hat = lti.m
            else:
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
                ourEnv.prime(data_size, lti.K0, 5, rng, lti)
                I_hat = ourEnv.I_hat
                m_hat = ourEnv.m_hat
            times[sim, i] = time.time() - start_time
            I_error[sim, i] = np.linalg.norm(I_hat - I_star)
            m_error[sim, i] = np.linalg.norm(m_hat - m_star)

    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    mean_color = [0.4940, 0.1840, 0.5560]
    # Function to add median line to boxplot
    def add_median_line(ax, boxplot, x_positions, color='blue', linestyle='-', label='Mean'):
        medians = [item.get_ydata()[0] for item in boxplot['medians']]  # Extract median values
        ax.plot(x_positions, medians, color=color, linestyle=linestyle, marker='o', label=label, linewidth=2)
        ax.legend()

    # Fixed x positions for boxplots
    x_positions = np.arange(1, len(step_vector) + 1)

    # Plot I_error
    plt.figure()
    boxplot_I = plt.boxplot(I_error, positions=x_positions, patch_artist=True, showfliers=False)
    for patch, color in zip(boxplot_I['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(x_positions, step_vector)
    add_median_line(plt.gca(), boxplot_I, x_positions, color=mean_color, linestyle='-', label='Mean $e_{I_b}$')
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Inertia Matrix Reconstruction Error ($e_{I_b}$)')
    plt.title('Inertia Matrix Reconstruction Error vs. Dataset Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/I_error_plot.png')
    plt.show()

    # Plot m_error
    plt.figure()
    boxplot_m = plt.boxplot(m_error, positions=x_positions, patch_artist=True, showfliers=False)
    for patch, color in zip(boxplot_m['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(x_positions, step_vector)
    add_median_line(plt.gca(), boxplot_m, x_positions, color=mean_color, linestyle='-', label='Mean $e_{m}$')
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Mass Reconstruction Error ($e_{m}$)')
    plt.title('Mass Reconstruction Error vs. Dataset Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/m_error_plot.png')
    plt.show()

    # Plot times (excluding the first value)
    plt.figure()
    boxplot_times = plt.boxplot(times[:, 1:], positions=x_positions[1:], patch_artist=True, showfliers=False)
    for patch, color in zip(boxplot_times['boxes'], colors[1:]):
        patch.set_facecolor(color)
    plt.xticks(x_positions[1:], step_vector[1:])
    add_median_line(plt.gca(), boxplot_times, x_positions[1:], color=mean_color, linestyle='-', label='Mean Time')
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time vs. Dataset Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/computation_time_plot.png')
    plt.show()


if __name__ == '__main__':
    MonteCarloSimulation()