import gym
from gym import spaces
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
from lib.hexapod_constant import *
from hexapod_control import *
from terrain import add_terrain
import matplotlib.pyplot as plt

HOME_POSITION = [0, 0, 0.1]
TARGET_POSITION = [-1, 1, 0.1]
is_training = True

class HexapodEnv(gym.Env):
    def __init__(self):
        # Initialize storage for plotting
        # Data storage
        self.reward = 0
        self.max_timesteps = 50
        self.timesteps = []
        self.rewards = []
        self.roll_input = []
        self.roll_robot = []
        self.pitch_input = []
        self.pitch_robot = []
        self.timestep = 0

        # Set up real-time plots with 3 subplots
        plt.ion() if not is_training else None # Interactive mode ON
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 10))

        # Reward plot
        self.line_reward, = self.ax1.plot([], [], 'b-', label="Reward")
        self.ax1.set_ylabel("Reward")
        self.ax1.legend()
        self.ax1.grid(True)

        # Roll plot (Fixed Y-axis from -20 to 20)
        self.line_roll_input, = self.ax2.plot([], [], 'r--', label="Roll Input (deg)")  # Dashed line
        self.line_roll_robot, = self.ax2.plot([], [], 'r-', label="Roll Robot (deg)")  # Solid line
        self.ax2.set_ylim(-25, 25)  # FIXED Y-AXIS
        self.ax2.set_ylabel("Roll (°)")
        self.ax2.legend()
        self.ax2.grid(True)

        # Pitch plot (Fixed Y-axis from -20 to 20)
        self.line_pitch_input, = self.ax3.plot([], [], 'g--', label="Pitch Input (deg)")  # Dashed line
        self.line_pitch_robot, = self.ax3.plot([], [], 'g-', label="Pitch Robot (deg)")  # Solid line
        self.ax3.set_ylim(-25, 25)  # FIXED Y-AXIS
        self.ax3.set_xlabel("Timestep")
        self.ax3.set_ylabel("Pitch (°)")
        self.ax3.legend()
        self.ax3.grid(True)
        
        super(HexapodEnv, self).__init__()

        self.start_time = time.time()
        # Set up PyBullet simulation
        self.client = p.connect(p.GUI if not is_training else p.DIRECT)

        self.robot = self.load_hexapod_model()

        # Define action space (vx, vy, vz, step_h, step_duration, phase)
        self.action_space = spaces.Box(low=np.array([-1, -1, 0]),
                                       high=np.array([1, 1, 0]),
                                       dtype=np.int32)

        # Define observation space (pos, pitch, roll, yaw)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

    def draw_circle(self, radius, position, color=[1, 0, 0]):
        # Remove previous circle lines if they exist
        if hasattr(self, "debug_line_ids") and self.debug_line_ids:
            for line_id in self.debug_line_ids:
                p.removeUserDebugItem(line_id)

        segments = 20  # Number of segments to form the circle
        points = []
        self.debug_line_ids = []  # Store IDs of debug lines

        # Generate points for the circle
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = position[0] + radius * math.cos(angle)
            y = position[1] + radius * math.sin(angle)
            points.append([x, y, position[2]])

        # Draw lines between points to form the circle
        for i in range(segments):
            line_id = p.addUserDebugLine(points[i], points[i + 1], lineColorRGB=color, lineWidth=2)
            self.debug_line_ids.append(line_id)  # Store the line ID

    def update_target(self):
        global TARGET_POSITION

        # Generate a random target position within a certain range
        TARGET_POSITION = [random.uniform(-2, 2), random.uniform(-2, 2), 0.1]

        # Draw the new target position
        self.draw_circle(0.1, TARGET_POSITION, [0, 1, 0])
        
    def load_hexapod_model(self):
        # Load your hexapod robot URDF here (make sure the model is available)
        # Load a ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane.urdf")
        terrain = add_terrain()

        p.changeDynamics(ground, -1, lateralFriction=1)
        p.changeDynamics(terrain, -1, lateralFriction=2.0)  # Higher friction

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)

        # Gambar lingkaran di posisi TARGET (merah)
        self.draw_circle(0.2, HOME_POSITION, [1, 0, 0])
        self.debug_line_ids = None
        # Gambar lingkaran di posisi HOME (hijau)
        self.draw_circle(0.1, TARGET_POSITION, [0, 1, 0])

        # Koordinat sudut kotak
        points = [(-2.5, -2.5, 0.1), (2.5, -2.5, 0.1), (2.5, 2.5, 0.1), (-2.5, 2.5, 0.1)]

        # Gambar garis kotak
        for i in range(4):
            p.addUserDebugLine(points[i], points[(i+1) % 4], [1, 0, 0])

        model_path = "models/hexapod.urdf"
        robot_id = p.loadURDF(model_path, basePosition=[0, 0, 0.5], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        return robot_id
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # Reset the robot to its initial position
        p.resetSimulation()
        self.robot = self.load_hexapod_model()
        p.stepSimulation()

        # Define initial state (pos, pitch, roll, yaw)
        initial_state = self.get_state()
        return np.array(initial_state, dtype=np.float32)

    def get_state(self):
        # Get robot position and orientation from PyBullet
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(orn)
        pitch, roll, yaw = euler
        return [pos[0], pos[1], pos[2], math.degrees(pitch), math.degrees(roll), math.degrees(yaw)]

    def step(self, action):
        # Apply action to the robot (action mapping to robot control)
        self.apply_action(action)

        # Get new state (feedback)
        state = self.get_state()
        p.addUserDebugLine([state[0], state[1], state[2]], TARGET_POSITION, lineColorRGB=[1, 0, 0], lineWidth=0.5, lifeTime=0.01)

        # Calculate reward
        self.reward = self.calculate_reward(state)

        # Done condition: Check if robot is stable or any other condition
        distance = math.sqrt((TARGET_POSITION[0] - state[0]) ** 2 + (TARGET_POSITION[1] - state[1]) ** 2)

        # Check if robot is flipped
        pitch, roll = state[3], state[4]
        if abs(pitch) > 90 or abs(roll) > 90:  # 90 degrees in radians
            print("Robot Flipped!!!")
            done = True
            self.reward -= 10  # Penalti jika terbalik
            self.start_time = time.time()
        elif abs(state[0])>2 or abs(state[1])>2:
            print("Robot Out!!!")
            done = True
            self.reward -= 5  # Penalti jika keluar
            self.start_time = time.time()
        else:
            done = False

        # print(f"distance: {distance}")
        if distance<0.1:
            self.update_target()
            done = True
            self.reward += 10
            self.start_time = time.time()
            
        return np.array(state, dtype=np.float32), self.reward, done, {}

    def apply_action(self, action):
        # Map action to robot's control (for now this is just a placeholder)
        t = time.time() - self.start_time

        # vx, vy, vz, step_h, step_duration, phase = [0.1, 0, 0, 0.05, 1, 2]

        # Heading alignment reward (robot should face the target)
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]  # Ensure yaw is correctly extracted
        x, y, z = pos

        target_angle = math.atan2(TARGET_POSITION[1] - y, TARGET_POSITION[0] - x)
        # Normalize angle difference to [-π, π]
        angle_diff = (target_angle - yaw) % (2 * math.pi) - math.pi
        vz = angle_diff * 0.1  # Increase the scaling factor for better respons

        # print(f"Target:{target_angle:.5f} yaw:{yaw:.5f} angle_diff:{angle_diff:.5f} vz:{vz:.5f}")

        # print(f"Action: {action}")
        self.r, self.p, self.y = (a * 20 for a in action)

        vx = 0.1
        vy = 0
        step_h, step_duration, phase = [0.1, 0.3, 2]
        body_movement = [vx, vy, vz]
        # print(action)

        leg_pos_body = body_kinematics([0,0,0], [self.r, self.p, self.y])

        for leg_index in range(6):  # Iterate over all 6 legs
            # Compute trajectory
            pos = generate_movement(t, phase, step_duration, step_h, body_movement, leg_index) #Pos akan menghasilkan array [x,y,z]
            leg_base = leg_pos_body[leg_index]
            leg_pos = (leg_base[0] + pos[0], leg_base[1] + pos[1], leg_base[2] + pos[2])
            
            # Control leg joint angles
            target_x, target_y, target_z = leg_pos

            # untuk kaki bagian kiri itu dikalikan negatif
            joint_index=leg_index * 3 + 1
            if joint_index>=10: 
                target_x*=-1
                target_y*=-1
            coxa_angle, femur_angle, tibia_angle = inverse_kinematics(target_x, target_y, target_z)

            p.setJointMotorControl2(self.robot, jointIndex=joint_index, controlMode=p.POSITION_CONTROL, targetPosition=coxa_angle)
            p.setJointMotorControl2(self.robot, jointIndex=joint_index + 1, controlMode=p.POSITION_CONTROL, targetPosition=femur_angle)
            p.setJointMotorControl2(self.robot, jointIndex=joint_index + 2, controlMode=p.POSITION_CONTROL, targetPosition=tibia_angle)

        p.stepSimulation()
        time.sleep(1 / 240) if not is_training else None

    def calculate_reward(self, state):
        w1 = 10  # Distance weight (higher priority)
        w2 = 5    # Stability weight (lower priority)

        # Extract current state values
        x, y, z, pitch, roll, yaw = state
        
        distance = -w1 * math.sqrt((TARGET_POSITION[0] - x) ** 2 + (TARGET_POSITION[1] - y) ** 2) + 10

        # Stability penalty (penalize large pitch/roll)
        stability_penalty = -w2 * (abs(pitch*0.1) + abs(roll*0.1))

        # Total reward
        reward = stability_penalty + distance

        # Clip the reward to a reasonable range
        # reward = np.clip(reward, -10, 10)

        # Debugging output
        # print(f"Reward: stability={stability_penalty:+06.2f}, total={reward:+06.2f} " f"d={distance/w1:+06.2f} r={self.r:+06.2f} p={self.p:+06.2f} y={self.y:+06.2f}")

        self.update_plot(self.reward, self.r, roll, self.p, pitch) if not is_training else None
        
        return reward
    
    def update_plot(self, reward, roll_input, roll_robot, pitch_input, pitch_robot):
        """Update the reward, roll, and pitch plots dynamically."""
        self.timesteps.append(self.timestep)
        self.rewards.append(reward)
        self.roll_input.append(roll_input)
        self.roll_robot.append(roll_robot)
        self.pitch_input.append(pitch_input)
        self.pitch_robot.append(pitch_robot)
        self.timestep += 1

        # Keep only the last 1000 timesteps
        if len(self.timesteps) > self.max_timesteps:
            self.timesteps = self.timesteps[-self.max_timesteps:]
            self.rewards = self.rewards[-self.max_timesteps:]
            self.roll_input = self.roll_input[-self.max_timesteps:]
            self.roll_robot = self.roll_robot[-self.max_timesteps:]
            self.pitch_input = self.pitch_input[-self.max_timesteps:]
            self.pitch_robot = self.pitch_robot[-self.max_timesteps:]

        # Update plot data
        self.line_reward.set_xdata(self.timesteps)
        self.line_reward.set_ydata(self.rewards)

        self.line_roll_input.set_xdata(self.timesteps)
        self.line_roll_input.set_ydata(self.roll_input)
        self.line_roll_robot.set_xdata(self.timesteps)
        self.line_roll_robot.set_ydata(self.roll_robot)

        self.line_pitch_input.set_xdata(self.timesteps)
        self.line_pitch_input.set_ydata(self.pitch_input)
        self.line_pitch_robot.set_xdata(self.timesteps)
        self.line_pitch_robot.set_ydata(self.pitch_robot)

        # Adjust x-axis limits dynamically
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        plt.draw()
        plt.pause(0.01)  # Small pause to refresh UI

    def render(self, mode='human'):
        # Implementasi rendering (optional)
        pass

    def close(self):
        p.disconnect()

