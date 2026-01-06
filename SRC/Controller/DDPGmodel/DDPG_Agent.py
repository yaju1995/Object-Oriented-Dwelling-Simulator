import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque

import numpy
import torch
print("NumPy version:", numpy.__version__)
print("PyTorch built with NumPy:", torch.__config__.show())

# from ..support.lib_config import CustomLogger

# Optional logger
try:
    from ..support.lib_config import CustomLogger

    logger = CustomLogger(False)


    def log(msg: str):
        logger.commandline(msg)
except Exception:
    def log(msg: str):
        pass


# ---------------------------
# Small MLP helper
# ---------------------------
def mlp(sizes, activation=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), activation=(nn.ReLU, nn.Tanh)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim],
                       activation=activation[0],
                       out_act=activation[1])  # Final activation controls output range

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), activation=(nn.ReLU, nn.Tanh)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1],
                       activation=activation[0],
                       out_act=nn.Identity)  # Final layer is a value → no activation

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)


# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Tuple[np.ndarray, float, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a: float, r: float, s2, done: bool):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# ---------------------------
# Config
# ---------------------------
@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float =1e-4
    buffer_capacity: int = 100_000
    hidden: Tuple[int, int] = (64, 64)
    activation: Tuple = (nn.ReLU, nn.Tanh)
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0


# --- DPG Agent ---
class DDPGAgent:
    def __init__(self, obs_dim, act_dim, max_action, cfg: DDPGConfig):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.actor = ActorNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation)
        self.actor_target = ActorNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation)
        self.critic_target = CriticNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.seed = cfg.seed

        self.max_action = max_action
        self.device = cfg.device
        self.tau = cfg.tau  # soft update factor

    def choose_action(self, state, noise_std=0.1):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        raw_action = self.actor(state_tensor).detach().numpy()[0]
        scaled_action = raw_action * self.max_action

        # Add noise for exploration
        noise = np.random.normal(0, noise_std, size=scaled_action.shape)
        return np.clip(scaled_action + noise, -self.max_action, self.max_action)

    def store_transition(self, state, action: float, reward: float, next_state, done: bool):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, done = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # ----- Critic update -----
        with torch.no_grad():
            # target_actions need to be scaled to match env bounds
            target_raw_actions = self.actor_target(next_states)
            target_actions = target_raw_actions * self.max_action

            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, target_actions)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor update -----
        raw_action = self.actor(states)  # ∈ [-1, 1]

        scaled_action = raw_action * self.max_action  # scale to env
        actor_loss = -self.critic(states, scaled_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft update target networks -----
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Agent loaded from {path}")


# --- Training Loop ---
def train_dpg():
    agent_cfg = DDPGConfig(gamma=0.99,
                           tau=0.995,
                           actor_lr=1e-3,
                           critic_lr=1e-3,
                           buffer_capacity=10000,
                           hidden=(64, 64),
                           activation=(nn.ReLU, nn.Tanh),
                           batch_size=128,
                           device="cuda" if torch.cuda.is_available() else "cpu",
                           seed=0)
    env = gym.make("Pendulum-v1", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    print(state_dim, action_dim, max_action)

    agent = DDPGAgent(state_dim, action_dim, max_action=2, cfg=agent_cfg)
    rewards = []

    for episode in range(1000):
        state, _ = env.reset()
        total_reward = 0

        for t in range(200):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train(batch_size=64)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")

    env.close()

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DPG Agent on Pendulum")
    plt.show()


if __name__ == "__main__":
    train_dpg()
