import time
import gymnasium as gym
import ale_py
from dqn import DQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Show Ai Runs
def main():
    #Set Rendermode
    visual_render = True
    rendering = ""

    if visual_render:
        rendering = "human"
    else:
        rendering = "rgb_array"
        
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    os.environ['SDL_AUDIODRIVER'] = 'dummy'


    # Basic ALE + Gymnasium Setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type='grayscale')
    # env = gym.wrappers.AtariPreprocessing(env, screen_size=128, grayscale_obs=True, noop_max=30, max_pool_frames=2)
    env = gym.wrappers.ResizeObservation(env, (100, 128))
    env = gym.wrappers.FrameStackObservation(env, 8)
    obs, info = env.reset()
    obs = obs[:, 10:68, :]

    # DQN Setup
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size, device).to(device)
    dqn.load_state_dict(torch.load("nn.pth"))
    dqn.eval()  # Set to evaluation mode

    # Main Loop: Runs till Terminated or Truncated
    start_time = time.time()
    episode_over = False
    total_steps = 0
    while not episode_over:
        action = 0
        with torch.no_grad():
            normalized_obs = obs.astype(np.float32) / 255.0
            state_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(device)
            q_values = dqn.forward(state_tensor)
            action = torch.argmax(q_values).item()
            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs[:, 10:68, :]
            total_steps += 1
            if total_steps % 300 == 0:
                x = state_tensor
                fig, ax = plt.subplots(len(dqn.conv_layers), 64, figsize=(100, 20))
                for i, layer in enumerate(dqn.conv_layers):
                    data = x.cpu().detach()[0]
                    
                    for j, x_i in enumerate(data):
                        if data.ndim == 3:
                            ax[i][j].imshow(x_i, cmap="gray")
                    for j in range(64):
                        ax[i][j].axis('off')
                    x = layer(x)
                print('saving')
                fig.tight_layout()
                fig.savefig("conv_layers.png")
                # fig.show()

        if visual_render:
            # time.sleep(.2)
            env.render()
        episode_over = terminated or truncated
        

    total_time = time.time() - start_time
    print(total_time)
    env.close()

if __name__ == "__main__":
    main()