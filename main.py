import time
import gymnasium as gym
import ale_py

def main():
    #Set Rendermode
    visual_render = False
    rendering = ""

    if visual_render:
        rendering = "human"
    else:
        rendering = "rgb_array"

    # Basic ALE + Gymnasium Setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type='ram')
    obs, info = env.reset()

    # Main Loop: Runs till Terminated or Truncated
    start_time = time.time()
    episodes = 10000
    episode_over = False
    for episode in range(episodes):
        if episode % 1000 == 0:
            print(f"Episode: {episode} / {episodes}")
        while not episode_over:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

    total_time = time.time() - start_time
    print(total_time)
    env.close()

if __name__ == "__main__":
    main()