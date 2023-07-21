from enum import Enum


class Task(str, Enum):
    """Environment tasks enumeration class."""

    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.

class CostType(str, Enum):
    POSITION = 'position'  # Position cost.
    POSITION_EULER = 'position_euler'  # Position and Euler angle cost.
    POSITION_QUATERNION = 'position_quaternion'  # Position and quaternion cost.


class DynamicsType(str, Enum):
    NORMAL_FIRST_ORDER = 'normal_first_order'  # Normal first order dynamics.
    NORMAL_SECOND_ORDER = 'normal_second_order'  # Normal second order dynamics.
    DIFF_FLAT = 'diff_flat'  # Differential flatness dynamics.


if __name__ == '__main__':
    print(Task.STABILIZATION)
    print(type(Task.STABILIZATION.value))