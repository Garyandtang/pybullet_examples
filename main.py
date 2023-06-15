import pybullet as p
import time
import pybullet_data
import numpy as np

def init_client():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)  # not real time
    client_inited = True
    return client_inited

class CartPole:
    def __init__(self, client_init=False):
        if not client_init:
            init_client()
        urdf_file = "cartpole.urdf"
        self.id = p.loadURDF(urdf_file, [0, 0, 0])
        self.nState = 4
        self.nControl = 1
        p.setTimeStep(0.02)
        self.state = np.zeros(self.nState)

    @property
    def get_id(self):
        return self.id

    @property
    def get_state(self):
        # [x, x_dot, theta, theta_dot]
        state = p.getJointState(self.id, 0)[0:2] + p.getJointState(self.id, 1)[0:2]
        self.state = np.array(state)
        return self.state

    def execute(self, force):
        p.setJointMotorControl2(self.id, 0, p.TORQUE_CONTROL, force=force)
        p.stepSimulation()
        # self.state = self.get_state()


if __name__ == '__main__':
    cart_pole = CartPole()

    for _ in range(100000):
        cart_pole.execute(10)
        time.sleep(0.02)



#
# # set up PyBullet physics simulation
# p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# # load cartpole model
# cartpole = p.loadURDF("cartpole.urdf", [0, 0, 0.5])
#
# # set up simulation parameters
# dt = 0.01   # time step
# max_steps = 1000   # maximum number of simulation steps
# theta_threshold = 0.5   # angle threshold for failure
#
# # initialize cartpole state variables
# x = 0.0   # cart position
# theta = 0.0   # pole angle
# x_dot = 0.0   # cart velocity
# theta_dot = 0.0   # pole angular velocity
#
#
#
#
#
# # control loop
# for i in range(max_steps):
#     # get current state of cartpole
#     x, _, _, x_dot, theta, _, theta_dot, _ = p.getJointState(cartpole, 0)[:8]
#
#     # check if pole has fallen too far
#     if abs(theta) > theta_threshold:
#         print("Pole fell over!")
#         break
#
#     # calculate control input (simple proportional controller)
#     Kp = 100   # proportional gain
#     u = -Kp*theta
#
#     # apply control input to cartpole
#     p.setJointMotorControl2(cartpole, 0, p.TORQUE_CONTROL, force=u)
#
#     # step simulation forward
#     p.stepSimulation()
#     time.sleep(dt)
#
# # clean up PyBullet simulation
# p.disconnect()