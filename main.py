import gymnasium as gym
import ale_py

# Basic ALE + Gymnasium Setup
gym.register_envs(ale_py)

env = gym.make("ALE/WizardOfWor-v5", render_mode="human")
obs, info = env.reset()

# Main Loop: Runs till Terminated or Truncated
episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()