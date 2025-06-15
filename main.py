import time
import gymnasium as gym
import ale_py
from dqn import DQN
import torch
import numpy as np

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

    # Basic ALE + Gymnasium Setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/WizardOfWor-v5", render_mode=rendering, obs_type='grayscale')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)
    obs, info = env.reset()

    # DQN Setup
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(state_size, action_size, device).to(device)
    dqn.load_state_dict(torch.load("nn.pth"))
    dqn.eval()  # Set to evaluation mode

    # Main Loop: Runs till Terminated or Truncated
    start_time = time.time()
    episode_over = False
    while not episode_over:
        action = 0
        with torch.no_grad():
            normalized_obs = obs.astype(np.float32) / 255.0
            state_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(device)
            q_values = dqn.forward(state_tensor)
            action = torch.argmax(q_values).item()
            obs, reward, terminated, truncated, info = env.step(action)

        if visual_render:
            time.sleep(.01)
            env.render()
        episode_over = terminated or truncated
        

    total_time = time.time() - start_time
    print(total_time)
    env.close()

if __name__ == "__main__":
    main()