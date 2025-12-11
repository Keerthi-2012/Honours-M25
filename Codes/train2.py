from robot_env2 import RobotEnv 
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

task_line_following = {
    'start': np.array([-4.0, 0.0]), 'end': np.array([4.0, 0.0]),
    'path': 'line', 'obstacles': False
}
task_circle_following_with_obstacles = {
    'start': np.array([-3.0, 0.0]), 'end': np.array([3.0, 0.0]),
    'path': 'circle', 'obstacles': True
}
task_direct_obstacle_avoidance = {
    'start': np.array([-4.0, -4.0]), 'end': np.array([4.0, 4.0]),
    'path': 'none', 'obstacles': True 
}

CURRENT_TASK_CONFIG = task_direct_obstacle_avoidance
ALGORITHM = SAC
TOTAL_TIMESTEPS = 200000

model_filename = f"{ALGORITHM.__name__}_final_task.zip"
train_env = make_vec_env(RobotEnv, n_envs=8, env_kwargs={'task_config': CURRENT_TASK_CONFIG})
model = ALGORITHM("MlpPolicy", train_env, verbose=1, device="cuda")

print(f"--- Starting Training for {ALGORITHM.__name__} ---")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(model_filename)
print(f"--- Training Complete. Model saved to {model_filename} ---")

print("\n--- Loading and Testing Trained Model ---")
eval_env = RobotEnv(render_mode="human", task_config=CURRENT_TASK_CONFIG)
loaded_model = ALGORITHM.load(model_filename)
for episode in range(5):
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