import numpy as np

## plot smoothed results, window size = 500

episode_reward = np.load("dqn_rewards.npy")
reward_per_step = np.load("dqn_rps.npy")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
fig, ax = plt.subplots(2)
ax[0].plot(np.convolve(episode_reward, np.ones(500)/500, mode="valid"))
ax[0].set_title("Cumulative Reward")
ax[0].set_xlabel("Episode")
ax[0].set_ylabel("Reward")
ax[0].grid(alpha=0.3)

ax[1].plot(np.convolve(reward_per_step, np.ones(500)/500, mode="valid"))
ax[1].set_title("Reward per Step")
ax[1].set_xlabel("Episode")
ax[1].set_ylabel("Reward")
ax[1].grid(alpha=0.3)

plt.savefig("dqn_smoothed.png")