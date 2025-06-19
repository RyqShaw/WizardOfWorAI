
# Wizard of Wor AI

A Project for our Artificial Intelligence Class: Building an AI that can beat the classic Atari 2600 game, Wizard of Wor.


## Authors

- [Sharyq Siddiqi](https://www.github.com/ryqshaw)
- [Shishir Pokhrel](https://www.github.com/pokhrel-sh)
- [Hakim Badmus](https://www.github.com/Hbadmus)
- [Sunil Williams](https://github.com/sunilwilliams4)

## Getting Started

Project running on Python 13\
To get started run `./setup.sh` on Linux/Mac, or `.\setup.bat` on Windows to install these Programs(Dependencies Included):\
ale-py==0.11.0\
gymnasium==1.1.1\
numpy==2.2.6\
pygame==2.6.1\
torch==2.7.0

when running `main.py`, you can set the model ran had on line 36 with `dqn.load_state_dict(torch.load("nn.pth"))`, where nn.pth is default model.\

when running `training.py`, and wanting to make your own model, you can change the factors and run this function `train(batch_size=64, gamma=0.999, epsilon=1, decay=.99999, max_episodes=10000, min_epsilon=0.1, max_episode_steps=18000, load_checkpoint = False)`. default train function is on line 208.\

Model is default saved as nn.pth

## Acknowledgements

 - [Arcade Learning Enviorment](https://ale.farama.org/)
 - [Gymnasium](https://gymnasium.farama.org/)
