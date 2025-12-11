from vo_moving import RobotEnv 
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os

task_line_following = {
    'start': np.array([-4.0, 0.0]), 'end': np.array([4.0, 0.0]),
    'path': 'line', 
    'obstacles': False
}
task_circle_following_with_obstacles = {
    'start': np.array([-3.0, 0.0]), 'end': np.array([3.0, 0.0]),
    'path': 'circle', 
    'obstacles': True,
    'num_obstacles': 3
}
task_direct_obstacle_avoidance = {
    'start': np.array([-4.0, -4.0]), 'end': np.array([4.0, 4.0]),
    'path': 'line', 
    'obstacles': True,
    'num_obstacles': 5
}

CURRENT_TASK_CONFIG = task_direct_obstacle_avoidance
ALGORITHM = SAC
TOTAL_TIMESTEPS = 2000000 

task_name = f"dynamic_{CURRENT_TASK_CONFIG['num_obstacles']}_obs"
model_filename = f"{ALGORITHM.__name__}_{task_name}.zip"
log_dir = os.path.join(".", "logs")
os.makedirs(log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=log_dir,
  name_prefix=f"{ALGORITHM.__name__}_model"
)

train_env = make_vec_env(RobotEnv, n_envs=8, env_kwargs={'task_config': CURRENT_TASK_CONFIG})

model = ALGORITHM(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    device="cuda", 
    tensorboard_log=log_dir,
    buffer_size=500000,
    learning_starts=10000
)

print(f"--- Starting Training for {ALGORITHM.__name__} on task: {task_name} ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=checkpoint_callback, 
    tb_log_name=task_name
)
model.save(model_filename)
print(f"--- Training Complete. Model saved to {model_filename} ---")
print(f"To view logs, run: tensorboard --logdir {log_dir}")

print("\n--- Loading and Testing Trained Model ---")
eval_env = RobotEnv(render_mode="human", task_config=CURRENT_TASK_CONFIG)
loaded_model = ALGORITHM.load(model_filename)

for episode in range(10): 
    obs, info = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

eval_env.close()