from enum import Enum


class Task(str, Enum):
    """Environment tasks enumeration class."""

    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.

class CostType(str, Enum):
    NAIVE = 'naive'  # Naive cost.
    POSITION = 'position'  # Position cost.
    POSITION_EULER = 'position_euler'  # Position and Euler angle cost.
    POSITION_QUATERNION = 'position_quaternion'  # Position and quaternion cost.


class DynamicsType(str, Enum):
    EULER_FIRST_ORDER = 'euler_first_order'  # first order dynamics with orientation in euler.
    EULER_SECOND_ORDER = 'euler_second_order'  # second order dynamics with orientation in euler.
    DIFF_FLAT = 'diff_flat'  # Differential flatness dynamics.


if __name__ == '__main__':
    print(Task.STABILIZATION)
    print(type(Task.STABILIZATION.value))