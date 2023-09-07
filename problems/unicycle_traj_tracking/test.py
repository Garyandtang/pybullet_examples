import numpy as np
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import scipy

def main():
    theta = np.linspace(-np.pi, np.pi, 1000)
    theta_d = -np.pi
    R_d = SO2(theta_d)
    orientation_error = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        R = SO2(theta[i])
        R_error = 0.5*(R_d.between(R) - R.between(R_d))

        orientation_error[i] = scipy.linalg.norm(R_error.coeffs())

    plt.figure()
    plt.plot(theta, orientation_error)
    plt.show()




if __name__ == '__main__':
    main()