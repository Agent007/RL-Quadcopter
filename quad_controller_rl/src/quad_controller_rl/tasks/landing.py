import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):

    def __init__(self):
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

        max_force = 20.0  # enough to land gently; 20.0 and greater will make it hover higher instead of lower
        min_force = 19.0
        max_torque = 20.0
        # limit minimum z-force to 0.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, min_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

        # Task-specific parameters
        self.max_duration = 30.0  # 5.0  # secs
        #self.max_error_position = 8.0  # distance units
        self.max_error_position = 12.0
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to land
        #self.weight_position = 1.0
        self.weight_position = 0.5  #0.0625
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        #self.weight_orientation = 0.3
        self.weight_orientation = 0.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity
        #self.weight_velocity = 1.0 - self.weight_position  # As the drone flies closer to the ground, put more importance on gentle landing.
        self.weight_velocity = 0.5  #0.9375

        self.last_timestamp = None
        self.last_position = None

        self.initial_height = 10.0

    def reset(self):
        #self.weight_position = 1.0
        #self.weight_velocity = 1.0 - self.weight_position
        
        x, y = np.random.normal(size=2)
        self.target_position[:2] = x, y
        z = self.initial_height + np.abs(np.random.normal(0.5, 0.1))
        p = [x, y, z]
        return Pose(
                position=Point(*p),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose, orientation, velocity only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        state = np.concatenate([position, orientation, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])  # Euclidean distance from target orientation quaternion (a better comparison may be needed)
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])
        #self.weight_position = np.clip(pose.position.z - self.target_position[2], 0.0, self.initial_height) / self.initial_height
        #self.weight_velocity = 1.0 - self.weight_position
        reward = -(self.weight_position * error_position + self.weight_orientation * error_orientation + self.weight_velocity * error_velocity) / (1.0 + (pose.position.z - self.target_position[2]))  # Increase penalty as drone gets closer to ground
        if error_position > self.max_error_position:
            print("strayed too far")
            #reward -= 5e3
            done = True
        elif timestamp > self.max_duration:
            print("timeout")
            #reward -= 5e3
            done = True
        elif pose.position.z < 0.2 and error_velocity > 2.0:
            print("approached ground too fast")
            #reward -= 5e3
            done = True
        elif velocity[2] > 0.2 and timestamp > 2.0:
            print("z velocity too high")
            reward -= 4e2
            done = True
        elif pose.position.z <= (self.target_position[2] + 0.1):
            reward += 4e2
            done = True
            print("landed")

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
