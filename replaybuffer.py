import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*transitions)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(next_states)
        )

    def __len__(self):
        return len(self.buffer)
    


# if __name__ == "__main__":
#     buffer = ReplayBuffer(capacity=5)

#     for i in range(5):
#         state = torch.tensor([i * 1.0, i * 2.0])
#         action = i % 2
#         reward = float(i)
#         next_state = torch.tensor([i * 1.5, i * 2.5])
#         buffer.add(state, action, reward, next_state)

#     print(f"Buffer length: {len(buffer)}")

#     states, actions, rewards, next_states = buffer.sample(4)

#     print("Sampled states:\n", states)
#     print("Sampled actions:\n", actions)
#     print("Sampled rewards:\n", rewards)
#     print("Sampled next_states:\n", next_states)

