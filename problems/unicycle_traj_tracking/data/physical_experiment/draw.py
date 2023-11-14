import numpy as np
import matplotlib.pyplot as plt
import os
from problems.unicycle_traj_tracking.data.butterfly_tracking.draw import load_data_and_save_figure

def main():
    gmpc_solver_time = np.load('Geomtric_model_predictive_control_store_solve_time.npy')
    nmpc_solver_time = np.load('Nonlinear_model_predictive_control_store_solve_time.npy')
    plt.figure()
    plt.boxplot([gmpc_solver_time, nmpc_solver_time], labels=['GMPC', 'NMPC'],showfliers=False, patch_artist=True )
    plt.ylabel('Solve time (s)')
    plt.tight_layout()
    plt.savefig('solve_time.jpg')
    plt.show()

if __name__ == '__main__':
    main()
