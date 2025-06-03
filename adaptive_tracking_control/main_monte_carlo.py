from data_driven_FBC import *
import os
import time
from planner.ref_traj_generator import TrajGenerator
from matplotlib import ticker
from adaptive_lqr.ts import TSStrategy
from adaptive_lqr.ofu import OFUStrategy

# Constants
FONT_SIZE = 16
LINE_WIDTH = 2
TRAJ_LENGTH = 300
DT = 0.02
NUM_SIMULATIONS = 5
NUM_ITERATIONS = 3
NUM_TRAINING_DATA = 1000


def generate_reference_trajectory():
    """Generate circular reference trajectory for the simulation."""
    ref_sysm = LTI()
    traj_config = {
        'type': TrajType.CIRCLE,
        'param': {
            'start_state': np.zeros((3,)),
            'linear_vel': ref_sysm.v,
            'angular_vel': ref_sysm.w,
            'nTraj': TRAJ_LENGTH - 1,
            'dt': DT
        }
    }
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    return ref_state[0, :], ref_state[1, :]  # ref_x, ref_y


def initialize_containers():
    """Initialize all data containers for the simulation."""
    containers = {
        # Parameter containers
        'K_containers': [np.zeros((NUM_SIMULATIONS, NUM_ITERATIONS + 1)) for _ in range(6)],
        'k_containers': [np.zeros((NUM_SIMULATIONS, NUM_ITERATIONS + 1)) for _ in range(2)],
        'r_container': np.zeros((NUM_SIMULATIONS, NUM_ITERATIONS + 1)),
        'l_container': np.zeros((NUM_SIMULATIONS, NUM_ITERATIONS + 1)),

        # Trajectory containers
        'x_trained': np.zeros((NUM_SIMULATIONS, TRAJ_LENGTH - 1)),
        'y_trained': np.zeros((NUM_SIMULATIONS, TRAJ_LENGTH - 1)),
        'x_init': np.zeros((NUM_SIMULATIONS, TRAJ_LENGTH - 1)),
        'y_init': np.zeros((NUM_SIMULATIONS, TRAJ_LENGTH - 1)),

        # Error containers
        'init_errors': {
            'x': np.zeros(NUM_SIMULATIONS),
            'y': np.zeros(NUM_SIMULATIONS)
        },
        'learned_errors': {
            'x': np.zeros(NUM_SIMULATIONS),
            'y': np.zeros(NUM_SIMULATIONS)
        },

        # Adaptive LQR containers
        'baseline_results': {
            'ts': {'r': np.zeros(NUM_SIMULATIONS), 'l': np.zeros(NUM_SIMULATIONS),
                   'x_error': np.zeros(NUM_SIMULATIONS), 'y_error': np.zeros(NUM_SIMULATIONS)},
            'ofu': {'r': np.zeros(NUM_SIMULATIONS), 'l': np.zeros(NUM_SIMULATIONS),
                    'x_error': np.zeros(NUM_SIMULATIONS), 'y_error': np.zeros(NUM_SIMULATIONS)}
        }
    }
    return containers


def setup_adaptive_strategy(strategy_type, lti, K_init, rng, num_data):
    """Setup and prime either TS or OFU adaptive strategy."""
    strategy_class = TSStrategy if strategy_type == 'ts' else OFUStrategy
    strategy = strategy_class(
        Q=lti.Q,
        R=lti.R,
        A_star=lti.A_ground_truth,
        B_star=lti.B_ground_truth,
        sigma_w=0,
        reg=1e-5,
        tau=500 if strategy_type == 'ts' else None,
        actual_error_multiplier=1,
        rls_lam=None
    )
    strategy.reset(rng)
    strategy.prime(num_data, K_init, 0.1, rng, lti)
    return strategy


def evaluate_strategy(strategy, lti, lti_eval, ref_x, ref_y, n_traj):
    """Evaluate the performance of a given strategy."""
    A_hat = strategy.estimated_A
    B_hat = strategy.estimated_B
    lti_eval.K0, lti_eval.k0 = lti.calculate_K_k(lti.A, B_hat)
    x_traj, y_traj, _, _ = evaluation(lti_eval, n_traj)
    x_error = np.linalg.norm(x_traj - ref_x)
    y_error = np.linalg.norm(y_traj - ref_y)
    return x_traj, y_traj, x_error, y_error, strategy.estimated_r, strategy.estimated_l


def run_learning_iteration(lti, iteration, containers, sim_idx):
    """Run a single learning iteration and update containers."""
    start_time = time.time()
    K, k, B, successful = learning(lti, 2)
    learning_time = time.time() - start_time

    if not successful:
        return False, learning_time

    r, l = calculate_r_l(B, lti.dt)

    # Update parameter containers
    containers['r_container'][sim_idx, iteration + 1] = r
    containers['l_container'][sim_idx, iteration + 1] = l

    # Update K matrix containers
    for i in range(6):
        containers['K_containers'][i][sim_idx, iteration + 1] = K[i // 3, i % 3]

    # Update k vector containers
    containers['k_containers'][0][sim_idx, iteration + 1] = k[0]
    containers['k_containers'][1][sim_idx, iteration + 1] = k[1]

    return True, learning_time


def create_boxplot(data, labels, ylabel, title, colors, save_path):
    """Create and save a boxplot with given parameters."""
    plt.figure()
    plt.grid(True)
    bplot = plt.boxplot(data, showfliers=False, patch_artist=True,
                        boxprops={'facecolor': 'none', 'alpha': 0.5})
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.title(title, fontsize=FONT_SIZE)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig(save_path)
    plt.show()


def save_trajectory_plot(x_data, y_data, xlabel, ylabel, title, save_path):
    """Create and save a trajectory plot."""
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=FONT_SIZE - 2)
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.plot(x_data.T, y_data.T, linewidth=LINE_WIDTH)
    plt.xlabel(xlabel, fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def save_parameter_plot(iterations, data, xlabel, ylabel, title, save_path):
    """Create and save a parameter evolution plot."""
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=FONT_SIZE - 2)
    plt.yticks(fontsize=FONT_SIZE - 2)
    plt.plot(iterations, data.T, linewidth=LINE_WIDTH)
    plt.xlabel(xlabel, fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def save_simulation_data(data_path, containers):
    """Save all simulation data to files."""
    np.save(os.path.join(data_path, 'r_container.npy'), containers['r_container'])
    np.save(os.path.join(data_path, 'l_container.npy'), containers['l_container'])
    np.save(os.path.join(data_path, 'x_init_container.npy'), containers['x_init'])
    np.save(os.path.join(data_path, 'y_init_container.npy'), containers['y_init'])
    np.save(os.path.join(data_path, 'x_trained_container.npy'), containers['x_trained'])
    np.save(os.path.join(data_path, 'y_trained_container.npy'), containers['y_trained'])


def print_statistics(containers):
    """Print summary statistics from the simulation."""
    # Print r statistics
    r_mean = np.mean(containers['r_container'][:, -1])
    r_std = np.std(containers['r_container'][:, -1])
    print(f"r_mean(ours): {r_mean:.4f}")
    print(f"r_std(ours): {r_std:.4f}")

    # Print l statistics
    l_mean = np.mean(containers['l_container'][:, -1])
    l_std = np.std(containers['l_container'][:, -1])
    print(f"l_mean(ours): {l_mean:.4f}")
    print(f"l_std(ours): {l_std:.4f}")


def MonteCarlo():
    """Main Monte Carlo simulation function."""
    # Setup reference trajectory and containers
    ref_x, ref_y = generate_reference_trajectory()
    containers = initialize_containers()

    # Create data directory
    dirpath = os.getcwd()
    data_path = os.path.join(dirpath, "data", "monte_carlo")
    os.makedirs(data_path, exist_ok=True)

    # Simulation parameters
    total_failures = 0
    avg_learning_time = 0
    lti_for_evaluation = LTI()
    rng = np.random

    # Main simulation loop
    for sim_idx in range(NUM_SIMULATIONS):
        lti = LTI()

        # Run OFU strategy
        ofu_strategy = setup_adaptive_strategy('ofu', lti, lti.K0, rng, NUM_TRAINING_DATA)
        (ofu_x, ofu_y,
         containers['baseline_results']['ofu']['x_error'][sim_idx],
         containers['baseline_results']['ofu']['y_error'][sim_idx],
         containers['baseline_results']['ofu']['r'][sim_idx],
         containers['baseline_results']['ofu']['l'][sim_idx]) = evaluate_strategy(
            ofu_strategy, lti, lti_for_evaluation, ref_x, ref_y, TRAJ_LENGTH)

        # Run TS strategy
        ts_strategy = setup_adaptive_strategy('ts', lti, lti.K0, rng, NUM_TRAINING_DATA)
        (ts_x, ts_y,
         containers['baseline_results']['ts']['x_error'][sim_idx],
         containers['baseline_results']['ts']['y_error'][sim_idx],
         containers['baseline_results']['ts']['r'][sim_idx],
         containers['baseline_results']['ts']['l'][sim_idx]) = evaluate_strategy(
            ts_strategy, lti, lti_for_evaluation, ref_x, ref_y, TRAJ_LENGTH)

        # Store initial trajectory and errors
        containers['x_init'][sim_idx, :], containers['y_init'][sim_idx, :], _, _ = evaluation(lti, TRAJ_LENGTH)
        containers['init_errors']['x'][sim_idx] = np.linalg.norm(containers['x_init'][sim_idx, :] - ref_x)
        containers['init_errors']['y'][sim_idx] = np.linalg.norm(containers['y_init'][sim_idx, :] - ref_y)

        # Store initial parameters
        containers['r_container'][sim_idx, 0], containers['l_container'][sim_idx, 0] = calculate_r_l(lti.B, lti.dt)
        init_K = lti.K_ini
        init_k = lti.k_ini

        for i in range(6):
            containers['K_containers'][i][sim_idx, 0] = init_K[i // 3, i % 3]
        containers['k_containers'][0][sim_idx, 0] = init_k[0]
        containers['k_containers'][1][sim_idx, 0] = init_k[1]

        # Learning iterations
        for iteration in range(NUM_ITERATIONS):
            successful, iter_time = run_learning_iteration(lti, iteration, containers, sim_idx)
            avg_learning_time += iter_time

            if not successful:
                print("Failed to learn")
                total_failures += 1
                sim_idx -= 1  # Repeat this simulation
                break

        # Store learned trajectory and errors
        containers['x_trained'][sim_idx, :], containers['y_trained'][sim_idx, :], _, _ = evaluation(lti, TRAJ_LENGTH)
        containers['learned_errors']['x'][sim_idx] = np.linalg.norm(containers['x_trained'][sim_idx, :] - ref_x)
        containers['learned_errors']['y'][sim_idx] = np.linalg.norm(containers['y_trained'][sim_idx, :] - ref_y)

    # Calculate average learning time
    avg_learning_time /= (NUM_SIMULATIONS * NUM_ITERATIONS)
    print(f"Average learning time: {avg_learning_time:.2f} seconds")

    # Plotting section
    colors = [[0.4940, 0.1840, 0.5560], [0.8500, 0.3250, 0.0980],
              [0.9290, 0.6940, 0.1250], [0.9290, 0.6940, 0.1250]]

    # Box plots for parameters
    create_boxplot(
        [containers['r_container'][:, 0], containers['baseline_results']['ofu']['r'],
         containers['baseline_results']['ts']['r'], containers['r_container'][:, -1]],
        ['initial', 'ofu', 'ts', 'our'],
        'length (m)',
        'radius r',
        colors,
        os.path.join(data_path, f"r_boxplot_{NUM_TRAINING_DATA}.jpg")
    )

    create_boxplot(
        [containers['l_container'][:, 0], containers['baseline_results']['ofu']['l'],
         containers['baseline_results']['ts']['l'], containers['l_container'][:, -1]],
        ['initial', 'ofu', 'ts', 'our'],
        'length (m)',
        'body length l',
        colors,
        os.path.join(data_path, f"l_boxplot_{NUM_TRAINING_DATA}.jpg")
    )

    # Box plots for errors
    create_boxplot(
        [containers['init_errors']['x'], containers['baseline_results']['ofu']['x_error'],
         containers['baseline_results']['ts']['x_error'], containers['learned_errors']['x']],
        ['initial', 'ofu', 'ts', 'our'],
        'Mean Square Error (m)',
        'x^p error',
        colors,
        os.path.join(data_path, f"x_error_boxplot_{NUM_TRAINING_DATA}.jpg")
    )

    create_boxplot(
        [containers['init_errors']['y'], containers['baseline_results']['ofu']['y_error'],
         containers['baseline_results']['ts']['y_error'], containers['learned_errors']['y']],
        ['initial', 'ofu', 'ts', 'our'],
        'Mean Square Error (m)',
        'y^p error',
        colors,
        os.path.join(data_path, f"y_error_boxplot_{NUM_TRAINING_DATA}.jpg")
    )

    # Trajectory plots
    save_trajectory_plot(
        containers['x_init'], containers['y_init'],
        "x (m)", "y (m)", "Initial Trajectories",
        os.path.join(data_path, "init_trajectory.jpg")
    )

    save_trajectory_plot(
        containers['x_trained'], containers['y_trained'],
        "x (m)", "y (m)", "Learned Trajectories",
        os.path.join(data_path, "trained_trajectory.jpg")
    )

    # Parameter evolution plots
    iterations = np.arange(0, NUM_ITERATIONS + 1, 1)
    save_parameter_plot(
        iterations, containers['r_container'],
        "iteration", "r (m)", "Radius Evolution",
        os.path.join(data_path, "trained_r.jpg")
    )

    save_parameter_plot(
        iterations, containers['l_container'],
        "iteration", "l (m)", "Body Length Evolution",
        os.path.join(data_path, "trained_l.jpg")
    )

    # Print statistics and save data
    print_statistics(containers)
    print(f"Total failures: {total_failures}")
    save_simulation_data(data_path, containers)


if __name__ == '__main__':
    MonteCarlo()