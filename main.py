import time
import gymnasium as gym
import ale_py
from dqn import DQN
import torch
import matplotlib.pyplot as plt


def main():
    #Set Rendermode
    visual_render = True
    rendering = ""

    if visual_render:
        rendering = "human"
    else:
        rendering = "rgb_array"

    # Basic ALE + Gymnasium Setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type='ram')
    obs, info = env.reset()


    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size)
    dqn.load_state_dict(torch.load("nn.pth"))
    dqn.eval()  # Set to evaluation mode

    # Main Loop: Runs till Terminated or Truncated
    start_time = time.time()
    episodes = 10000
    episode_over = False
    for episode in range(episodes):
        if episode % 1000 == 0:
            print(f"Episode: {episode} / {episodes}")
        while not episode_over:
            action = 0
            with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = dqn.forward(state_tensor)
                    action = torch.argmax(q_values).item()
            obs, reward, terminated, truncated, info = env.step(action)

            if visual_render:
                time.sleep(.01)
                frame = env.render()
                # plt.imshow(frame)
                # plt.show()
            episode_over = terminated or truncated

    total_time = time.time() - start_time
    print(total_time)
    env.close()

if __name__ == "__main__":
    main()