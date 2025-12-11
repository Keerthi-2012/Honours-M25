import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RobotEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, task_config=None):
        super(RobotEnv, self).__init__()

        if task_config is None:
            task_config = {
                'start': np.array([-4.0, 0.0]),
                'end': np.array([4.0, 0.0]),
                'path': 'line',
                'obstacles': True,
                'num_obstacles': 4
            }

        self.config = task_config
        self.start_point = self.config['start'].astype(np.float32)
        self.goal_point = self.config['end'].astype(np.float32)

        self.dt = 0.1
        self.max_episode_steps = 700
        self.robot_radius = 0.15
        self.goal_radius = 0.2
        self.goal_reach_threshold = self.robot_radius + self.goal_radius

        self.w_progress = 20.0
        self.w_path = 2.0
        self.w_obstacle = 0.3
        self.w_vo = 10.0 

        self.max_obstacles = 10 
        self.num_obstacles = 0
        if self.config['obstacles']:
            self.num_obstacles = self.config.get('num_obstacles', 4)
            
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(-3.0, 3.0, size=2)
            vel = np.random.uniform(-0.5, 0.5, size=2)
            self.obstacles.append({
                'pos': pos.astype(np.float32),
                'vel': vel.astype(np.float32),
                'radius': 0.4
            })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        
        obs_dim = 2 + 3 + (self.max_obstacles * 5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.fig, self.ax = None, None

    def _rotate(self, vec, angle):
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        return rot_matrix @ vec

    def _get_obs(self):
        obs_parts = []
        theta = self.state[2]
        robot_pos = self.state[:2]

        obs_parts.extend([np.sin(theta), np.cos(theta)])

        goal_rel_world = self.goal_point - robot_pos
        goal_rel_robot = self._rotate(goal_rel_world, -theta)
        obs_parts.extend(goal_rel_robot)
        obs_parts.append(np.linalg.norm(goal_rel_world)) 

        obstacle_info = []
        for i in range(self.num_obstacles):
            obs = self.obstacles[i]
            pos_rel_world = obs['pos'] - robot_pos
            vel_rel_world = obs['vel']
            
            pos_rel_robot = self._rotate(pos_rel_world, -theta)
            vel_rel_robot = self._rotate(vel_rel_world, -theta)
            
            obstacle_info.extend(pos_rel_robot)
            obstacle_info.extend(vel_rel_robot)
            obstacle_info.append(np.linalg.norm(pos_rel_world)) 

        num_filled = self.num_obstacles
        padding = np.zeros((self.max_obstacles - num_filled) * 5, dtype=np.float32)
        
        obs_parts.extend(obstacle_info)
        obs_parts.extend(padding)
        
        return np.array(obs_parts, dtype=np.float32)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs['pos'] += obs['vel'] * self.dt
            obs['vel'] += np.random.uniform(-0.05, 0.05, size=2)
            obs['vel'] = np.clip(obs['vel'], -0.7, 0.7)
            for i in range(2):
                if obs['pos'][i] < -4.5 or obs['pos'][i] > 4.5:
                    obs['vel'][i] *= -1.0
                    obs['pos'][i] = np.clip(obs['pos'][i], -4.5, 4.5)

    def step(self, action):
        self.step_count += 1
        v = action[0] * 1.0
        omega = action[1] * np.pi
        x, y, theta = self.state

        if self.config['obstacles']:
            self._update_obstacles()

       
        vo_penalty = 0.0

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = np.arctan2(np.sin(theta + omega * self.dt), np.cos(theta + omega * self.dt))
        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)

        dist_goal = np.linalg.norm(self.state[:2] - self.goal_point)
        progress = self.prev_dist_to_goal - dist_goal
        self.prev_dist_to_goal = dist_goal
        reward = self.w_progress * progress - 0.1 

        if self.config['path'] == 'line':
            line_vec = self.goal_point - self.start_point
            robot_vec = self.state[:2] - self.start_point

            line_norm = np.linalg.norm(line_vec)
            if line_norm > 1e-6:
                cte = np.abs(np.cross(line_vec, robot_vec)) / line_norm
                reward -= self.w_path * cte
        elif self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius = np.linalg.norm(self.start_point - center)
            cte = abs(np.linalg.norm(self.state[:2] - center) - radius)
            reward -= self.w_path * cte

        if self.config['obstacles']:
            all_dists = [np.linalg.norm(self.state[:2] - obs['pos']) - (obs['radius'] + self.robot_radius)
                         for obs in self.obstacles]
            d_min = min(all_dists)

            if d_min <= 0: 
               
                return self._get_obs(), -100, True, False, {}

            reward -= self.w_obstacle * np.exp(-d_min * 2.0)

        terminated = dist_goal < self.goal_reach_threshold
        if terminated:
            reward += 100
        truncated = self.step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        start_theta = 0.0

        if self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius_vector = self.start_point - center
            start_theta = np.arctan2(radius_vector[1], radius_vector[0]) + np.pi / 2
        else:
            start_theta = np.arctan2(self.goal_point[1] - self.start_point[1], 
                                     self.goal_point[0] - self.start_point[0])

        self.state = np.array([self.start_point[0], self.start_point[1], start_theta], dtype=np.float32)
        self.prev_dist_to_goal = np.linalg.norm(self.state[:2] - self.goal_point)

        if self.config['obstacles']:
            for obs in self.obstacles:
                obs['pos'] = np.random.uniform(-3.0, 3.0, size=2)
                
                while (np.linalg.norm(obs['pos'] - self.start_point) < 1.0 or 
                       np.linalg.norm(obs['pos'] - self.goal_point) < 1.0):
                    obs['pos'] = np.random.uniform(-3.0, 3.0, size=2)
                obs['vel'] = np.random.uniform(-0.5, 0.5, size=2)

        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def _render_frame(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()

        self.ax.add_artist(plt.Circle(self.start_point, 0.2, color='b'))
        self.ax.add_artist(plt.Circle(self.goal_point, self.goal_radius, color='g'))

        if self.config['path'] == 'line':
            self.ax.plot([self.start_point[0], self.goal_point[0]],
                         [self.start_point[1], self.goal_point[1]], 'g--', alpha=0.5)
        elif self.config['path'] == 'circle':
            center = (self.start_point + self.goal_point) / 2
            radius = np.linalg.norm(self.start_point - center)
            self.ax.add_artist(plt.Circle(center, radius, color='g', fill=False, linestyle='--', alpha=0.5))

        if self.config['obstacles']:
            for obs in self.obstacles:
                self.ax.add_artist(plt.Circle(obs['pos'], obs['radius'], color='r'))

        x, y, theta = self.state
        self.ax.add_artist(plt.Circle((x, y), self.robot_radius, color='c', fill=True, alpha=0.7))
        self.ax.arrow(x, y, 0.3 * np.cos(theta), 0.3 * np.sin(theta),
                      head_width=0.1, head_length=0.15, fc='k', ec='k')

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title(f"Step: {self.step_count}")
        self.fig.canvas.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None