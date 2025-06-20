import time
import gymnasium as gym
import ale_py
import random
import torch
from replaybuffer import ReplayBuffer
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
import os
from ale_py import ALEInterface
from torch.amp import autocast, GradScaler

# setup
gym.register_envs(ale_py)
env = gym.make("ALE/WizardOfWor-v5", obs_type="grayscale")

env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.FrameStackObservation(env, 4)

ale = env.unwrapped.ale

# Get env info
obs, info = env.reset()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

state_size = obs.shape[0]
action_size = env.action_space.n

print_statement_interval = 100
checkpoint_interval = 1000
model_path = "nn.pth"

# Just penelizes doing nothing for now
def calculate_reward(base_reward, action):
    
    reward = base_reward
    if action == 0:  # No-op
        reward -= 1.0

    #put the shooting peneties here
    # no shooting penelties right now
    return reward

# training with 64 concurrent episodes, uses DQN and Replay Buffer Implementation
def train(batch_size=64, gamma=0.999, epsilon=1, decay=.99999, max_episodes=10000, min_epsilon=0.1, max_episode_steps=18000, load_checkpoint = False):
    # init replay buffer with 10k size
    episode_rewards = []
    replay_buffer = ReplayBuffer(10000) 

    policy_nn = DQN(state_size, env.action_space.n, device).to(device)
    target_nn = DQN(state_size, env.action_space.n, device).to(device)
    min_replay_size = 5000
    #print("Initialized DQN", dqn.parameters())
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=0.00025)
    scaler = GradScaler()
    #print("Initialized Optimizer")
    episodes_done = 0
    total_steps = 0

    if os.path.exists("checkpoint.path") and load_checkpoint:
        print("Loading Checkpoint")
        checkpoint = torch.load("checkpoint.path")
        episodes_done = checkpoint['episode']
        policy_nn.load_state_dict(checkpoint['policy_state_dict'])
        target_nn.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
    
    # training
    print("Starting Training:")
    for episode in range(episodes_done, max_episodes):
        obs, info = env.reset()
        episode_over = False
        total_reward = 0
        episode_steps = 0
        current_lives = ale.lives()

        if episode % print_statement_interval == 0:
            print(f"Episode {episode} / {max_episodes}")

        while not episode_over:
            normalized_obs = obs.astype(np.float32) / 255.0
            # epsilon greedy with random 
            action = 0
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(device)
                    """ print(f"Training input shape: {state_tensor.shape}") 
                    print(f"Training Q-values: {policy_nn.forward(state_tensor)}")  """
                    q_values = policy_nn.forward(state_tensor)
                    action = torch.argmax(q_values).item()

            new_obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_steps += 1
            episode_over = terminated or truncated

            if ale.lives() < current_lives:
                reward -= 10.0
                current_lives = ale.lives()
                
            reward = calculate_reward(reward, action)

            # scales reward
            clipped_reward = np.clip(reward, -10, 10)
            
            # adds to buffer
            normalized_new_obs = np.array(new_obs, dtype=np.float32) / 255.0
           
            # store the experience in the replay buffer
            replay_buffer.add(normalized_obs, action, clipped_reward, normalized_new_obs)
            
            new_obs = np.array(new_obs)
            obs = new_obs
            total_reward += reward

            # train dqn and store state, action, rewards and next state.
            if len(replay_buffer) >= min_replay_size and total_steps % 4 == 0:
                
                
                states, actions, rewards, next_states = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)

                # #print the action information every 100 steps
                # if total_steps % 100 == 0:
                #     print(f"Current batch rewards: {rewards[:5]}")  
                #     print(f"Current batch actions: {actions[:5]}")  
                #     print(f"Episode total reward so far: {total_reward:.2f}")
                #     print(f"Steps in episode: {episode_steps}")                

                with autocast(device):
                    q_vals = policy_nn.forward(states).gather(1, actions.unsqueeze(1))
                    
                    # q-learning, get the max from the first state and make that the target
                    with torch.no_grad():
                        next_actions = policy_nn(next_states).argmax(1)
                        next_q_vals = target_nn(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                        target_q_vals = rewards + (gamma * next_q_vals)

                    loss = mse(q_vals.squeeze(), target_q_vals)
                
                # update gradiant
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # gradient clipping to prevent instability in training
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), max_norm=10.0)
                
                # free mem
                del states, next_states, q_vals, target_q_vals, loss
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # free mem
                if total_steps % 100 == 0:
                    torch.cuda.empty_cache()
                # Update target every 1000 steps AI takes
                if total_steps % 1000 == 0:
                    target_nn.load_state_dict(policy_nn.state_dict())
            
            episode_over = truncated or terminated
            # check for episode limits
            if episode_steps >= max_episode_steps:
                episode_over = True
        episode_rewards.append(total_reward)
        #print(total_reward)
        
        epsilon = max(min_epsilon, epsilon * decay)
        
        # Update Target NN and checkpoint
        if episode % checkpoint_interval == 0 and episode != 0:
            print("Checkpoint!")
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy_nn.state_dict(),
                'target_state_dict': target_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }
            torch.save(checkpoint, "checkpoint.path")
    # show the scatter plot
    plt.scatter(range(len(episode_rewards)), episode_rewards)
    plt.title("Episode Rewards Over Time: 1000 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    return policy_nn

start_time = time.time()
dqn = train(max_episodes=1000, load_checkpoint = False)

#run this on sharyq gpu when confident it all works
#dqn = train(batch_size=256, max_episodes=15000, load_checkpoint = False)
print(f"Training time: {time.time() - start_time}")

torch.save(dqn.state_dict(), model_path)
