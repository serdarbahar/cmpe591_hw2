import numpy as np
from homework2 import Hw2Env
import torch
from dqn import DQN
from replay_memory import ReplayMemory
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm

GAMMA=0.99
EPS=1.0
EPS_DECAY=0.995
EPS_DECAY_ITER = 1 # per episode
MIN_EPSILON=0.05
LR=1e-4
BATCH_SIZE=64
UPDATE_FREQ=10
TARGET_NETWORK_UPDATE_FREQ=200
MEMORY_SIZE = 100000
N_ACTIONS = 8
N_STATE = 6
NUM_EPISODES = 1000

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def train():
    policy_net = DQN(N_STATE, N_ACTIONS).to(device)
    target_net = DQN(N_STATE, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    episode_reward = []
    reward_per_step = []

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    steps_done = 0

    for i_episode in tqdm(range(NUM_EPISODES), desc="Training"):
        env.reset()
        state = torch.tensor(env.high_level_state(), dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        cumulative_reward = 0.0

        for t in range(env._max_timesteps):
            action = utils.select_action(policy_net, state, EPS, N_ACTIONS)
            obs, reward, is_terminal, is_truncated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = is_terminal or is_truncated

            if is_terminal:
                next_state = None
            else:
                next_state = torch.tensor(env.high_level_state(), dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action.to(device), next_state, reward)

            state = next_state

            # update policy net every UPDATE_FREQ steps
            if steps_done % UPDATE_FREQ == 0:
                utils.optimize_model(policy_net, target_net, memory, optimizer, device, BATCH_SIZE, GAMMA)

            # update target net every TARGET_NETWORK_UPDATE_FREQ steps
            if steps_done % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            cumulative_reward += reward.item()

            steps_done += 1

            if done:
                break
        
        # decay epsilon every EPS_DECAY_ITER episodes
        if steps_done % EPS_DECAY_ITER == 0:
            EPS = max(MIN_EPSILON, EPS*EPS_DECAY)
        
        episode_reward.append(cumulative_reward)
        if t>0:
            reward_per_step.append(cumulative_reward/t)
        else:
            reward_per_step.append(0)

        tqdm.write(f"Episode={i_episode}, Reward={cumulative_reward}, RPS={cumulative_reward/t if t>0 else 0}, EPS={EPS}")
        #print(f"Episode={i_episode}, Reward={cumulative_reward}, RPS={cumulative_reward/t if t>0 else 0}, EPS={EPS}")

    # plot figure with 2 subplots, cumulative reward and RPS
    # make figure bigger
    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots(2)
    ax[0].plot(episode_reward)_
    ax[0].set_title("Cumulative Reward")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].grid(alpha=0.3)

    ax[1].plot(reward_per_step)
    ax[1].set_title("Reward per Step")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Reward")
    ax[1].grid(alpha=0.3)

    plt.savefig("dqn.png")

    ## also save rewards as npy file
    np.save("dqn_rewards.npy", np.array(episode_reward))
    np.save("dqn_rps.npy", np.array(reward_per_step))

    ## save the model
    torch.save(policy_net.state_dict(), "dqn.pth")

def test():
    policy_net = DQN(N_STATE, N_ACTIONS).to(device)
    policy_net.load_state_dict(torch.load("dqn.pth"))

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
    env.reset()

    state = torch.tensor(env.high_level_state(), dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        action = policy_net(state).argmax().unsqueeze(0)
        obs, reward, is_terminal, is_truncated = env.step(action.item())
        state = torch.tensor(env.high_level_state(), dtype=torch.float32, device=device).unsqueeze(0)
        done = is_terminal or is_truncated