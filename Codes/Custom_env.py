import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RobotEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, task_config=None):
        super(RobotEnv, self).__init__()
        
        # Default Task Configuration
        if task_config is None:
            task_config = {
                'start': np.array([-4.0, 0.0]),
                'end': np.array([4.0, 0.0]),
                'path': 'line',
                'obstacles': False
            }
        self.config = task_config
        self.start_point = self.config['start']
        self.goal_point = self.config['end']

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_low = np.array([-5.0, -5.0, -np.pi], dtype=np.float32)
        obs_high = np.array([5.0, 5.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Environment & Reward Parameters
        self.dt = 0.1
        self.max_episode_steps = 700
        self.w_progress = 15.0
        self.w_path = 2.0
        self.w_obstacle = 1.0
        
        # Obstacle Setup
        if self.config['obstacles']:
            self.obstacles = [
                {'pos': np.array([0.0, 0.0], dtype=np.float32), 'radius': 0.5},
                {'pos': np.array([2.0, -1.0], dtype=np.float32), 'radius': 0.5}
            ]
        
        # Visualization
        self.render_mode = render_mode
        self.fig, self.ax = None, None

    def step(self, action):
        self.step_count += 1
        v = action[0] * 1.0; omega = action[1] * np.pi
        x, y, theta = self.state
        x_new, y_new = x + v * np.cos(theta) * self.dt, y + v * np.sin(theta) * self.dt
        theta_new = np.arctan2(np.sin(theta + omega * self.dt), np.cos(theta + omega * self.dt))
        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)
        
        # Reward Calculation
        dist_goal = np.linalg.norm(self.state[:2] - self.goal_point)
        progress = self.prev_dist_to_goal - dist_goal
        self.prev_dist_to_goal = dist_goal
        reward = self.w_progress * progress

        if self.config['path'] == 'line':
            line_vec = self.goal_point - self.start_point
            robot_vec = self.state[:2] - self.start_point
            cte = np.abs(np.cross(line_vec, robot_vec)) / np.linalg.norm(line_vec)
            reward -= self.w_path * cte
        elif self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius = np.linalg.norm(self.start_point - center)
            cte = abs(np.linalg.norm(self.state[:2] - center) - radius)
            reward -= self.w_path * cte

        if self.config['obstacles']:
            d_min = min(np.linalg.norm(self.state[:2] - obs['pos']) - obs['radius'] for obs in self.obstacles)
            if d_min <= 0: return self.state, -100, True, False, {}
            reward -= self.w_obstacle * np.exp(-d_min)

        terminated = dist_goal < 0.3
        if terminated: reward += 100
        truncated = self.step_count >= self.max_episode_steps
        
        if self.render_mode == "human": self._render_frame()
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        start_theta = 0.0
        if self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius_vector = self.start_point - center
            start_theta = np.arctan2(radius_vector[1], radius_vector[0]) + np.pi / 2
        
        self.state = np.array([self.start_point[0], self.start_point[1], start_theta], dtype=np.float32)
        self.prev_dist_to_goal = np.linalg.norm(self.state[:2] - self.goal_point)
        
        if self.render_mode == "human": self._render_frame()
        return self.state, {}

    def _render_frame(self):
        if self.fig is None:
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        
        self.ax.add_artist(plt.Circle(self.start_point, 0.2, color='b'))
        self.ax.add_artist(plt.Circle(self.goal_point, 0.2, color='g'))
        
        if self.config['path'] == 'line':
            self.ax.plot([self.start_point[0], self.goal_point[0]], [self.start_point[1], self.goal_point[1]], 'g--')
        elif self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius = np.linalg.norm(self.start_point - center)
            self.ax.add_artist(plt.Circle(center, radius, color='g', fill=False, linestyle='--'))

        if self.config['obstacles']:
            for obs in self.obstacles: self.ax.add_artist(plt.Circle(obs['pos'], obs['radius'], color='r'))
        
        x, y, theta = self.state
        self.ax.arrow(x, y, 0.3*np.cos(theta), 0.3*np.sin(theta), head_width=0.1, head_length=0.15, fc='b', ec='b')
        self.ax.set_xlim(-5, 5); self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal'); self.ax.grid(True)
        self.ax.set_title(f"Step: {self.step_count}")
        self.fig.canvas.draw(); plt.pause(0.001)

    def close(self):
        if self.fig is not None: plt.close(self.fig); self.fig, self.ax = None, None