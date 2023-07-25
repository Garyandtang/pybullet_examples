

# Control Framework with Pybullet and CasADi
This repo is a control framework for robot control with pybullet. It is based on the [pybullet quickstart guide](https://pybullet.org/wordpress/) and the [casadi quickstart guide]()

## Installation


## Framework Structure
The framework is structured in the following way:
1. Environment: pybullet based physical environment with a robot
   * the robot is described by a urdf file
   * the environment is described by a python class 
   * the simulation is set to be non-realtime
   * to control the robot, use the `step()` api with velocity/toque commands
   * the output of the `step()` api is the robot state
2. Symbolic Model: casadi based symbolic model of the robot 
   * currently, the symbolic model is generated from the config dict
   * todo: generate the symbolic model from the urdf file
   * the symbolic model is used to generate the kinematics of the robot
     * `symbolic`: contains
       * ode function
       * jacobian of the ode function
       * cost function (todo: decouple cost from model)
3. Problem: 
    * the problem is defined by a class
    * the class contains
      * the symbolic model
      * the initial state
      * the goal state
      * the cost function
      * the constraints
      * the solver options

4. Controller: 
    * the controller is defined by a class
    * input: problem
    * output: control sequence