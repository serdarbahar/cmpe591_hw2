## test DQN in simulation

import torch
from homework2 import Hw2Env
from dqn import DQN

device = torch.device("cpu")
N_ACTIONS = 8
N_STATE = 6
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

env.close()