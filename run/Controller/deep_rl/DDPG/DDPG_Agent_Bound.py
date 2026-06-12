from dataclasses import dataclass
from typing import  Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from run.Controller.deep_rl.DDPG.DDPGAgent import DDPGConfig, DDPGAgent

@dataclass
class DDPGBoundConfig(DDPGConfig):
    name: str = "DRL_DDPG_Bound"
    algorithm: str = "DDPG"

    action_type: str = "continuous"

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-4
    critic_lr: float = 1e-4

    buffer_capacity: int = 100_000
    hidden = (128, 128)
    activation = (nn.ReLU, nn.Tanh)

    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0

    a_max: Optional[float] = 1.0
    a_min: Optional[float] = -1.0

    def generate_model_name(self) -> str:
        name_parts = [
            self.name,
            f"G{int(self.gamma * 100)}",
            f"T{str(self.tau).replace('.', '')}",
            f"ALR{self.actor_lr:.0e}",
            f"CLR{self.critic_lr:.0e}",
            f"H{'x'.join(map(str, self.hidden))}",
            f"B{self.batch_size}",
            f"Buf{self.buffer_capacity // 1000}k",
            self.device.upper(),
        ]
        return "-".join(name_parts)

    def generate_config_text(self) -> str:
        activation_names = ", ".join([act.__name__ for act in self.activation])
        bound_name = self.bound_fn.__name__ if self.bound_fn is not None else "None"

        return f"""Model Name: {self.generate_model_name()}

Configuration Summary:
-----------------------
Name            : {self.name}
Algorithm       : {self.algorithm}
Gamma (γ)       : {self.gamma}
Tau (τ)         : {self.tau}
Actor LR        : {self.actor_lr}
Critic LR       : {self.critic_lr}
Hidden Layers   : {self.hidden}
Activations     : ({activation_names})
Observation Dim : {self.obs_dim}
Action Dim      : {self.action_dim}
Action Type     : {self.action_type}
Batch Size      : {self.batch_size}
Buffer Capacity : {self.buffer_capacity}
Device          : {self.device.upper()}
Seed            : {self.seed}
Action Min      : {self.a_min}
Action Max      : {self.a_max}
Bound Function  : {bound_name}
"""


class DDPGBoundAgent(DDPGAgent):
    def __init__(self, config: DDPGBoundConfig):
        super().__init__(config)

        self.bound_fn = self.config.bound_fn

    def get_action_bounds(self, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.bound_fn is None:
            self.bound_fn = self.default_bound_fn
        if not callable(self.bound_fn):
            raise ValueError("bound_fn must be callable(state)->(a_min,a_max)")
        return self.bound_fn(state)

    def default_bound_fn(self, state):
        # Always return static range (no state logic)
        return np.array(self.config.a_min), np.array(self.config.a_max)

    def scale_action(self, raw_action: np.ndarray, a_min: torch.tensor, a_max: torch.tensor) -> np.ndarray:
        return a_min + 0.5 * (raw_action + 1.0) * (a_max - a_min)

    @torch.no_grad()
    def get_action(self, state, noise_std: float = 0.1):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ra = self.actor(s).cpu().numpy()[0]  # ra raw action

        a_min, a_max = self.get_action_bounds(state)

        a = self.scale_action(ra, a_min, a_max)
        if noise_std > 0:
            a = a + self.rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, a_min, a_max)

    def store_transition(self, state, action, reward: float, next_state, done: bool):
        # print(reward)
        self.collector.add(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.config.batch_size:
            return None
        # print(f'Updating!{len(self.buffer)} <{batch_size}')
        states, actions, R, next_states, dones, gamma_pows = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        R = torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_pows = torch.tensor(gamma_pows, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Get dynamic bounds for each next_state (batched)
        # ns_np = next_states.detach().cpu().numpy()
        a_min_list, a_max_list = [], []
        for ns in next_states.detach().cpu().numpy():
            amin, amax = self.get_action_bounds(ns)
            a_min_list.append(amin)
            a_max_list.append(amax)

        a_min_tensor = torch.tensor(a_min_list, dtype=torch.float32, device=self.device)
        a_max_tensor = torch.tensor(a_max_list, dtype=torch.float32, device=self.device)

        # Works for both:
        # - n-step: gamma_pows=gamma^k and dones maybe 0
        # - MC: gamma_pows=0 and dones=1 => y = R (R is G)
        with torch.no_grad():
            next_raw_actions = self.actor_target(next_states)
            next_actions = self.scale_action(next_raw_actions, a_min_tensor, a_max_tensor)
            # next_actions = a_min_tensor + 0.5 * (next_raw_actions + 1.0) * (a_max_tensor - a_min_tensor)
            target_q = self.critic_target(next_states, next_actions)
            y = R + gamma_pows * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Get dynamic bounds for current states
        a_min_list, a_max_list = [], []
        for s in states.detach().cpu().numpy():
            amin, amax = self.get_action_bounds(s)
            a_min_list.append(amin)
            a_max_list.append(amax)

        a_min_tensor = torch.tensor(a_min_list, dtype=torch.float32, device=self.device)
        a_max_tensor = torch.tensor(a_max_list, dtype=torch.float32, device=self.device)
        # ----- Actor update -----
        raw_actor_actions = self.actor(states)
        actor_actions = self.scale_action(raw_actor_actions, a_min_tensor, a_max_tensor)

        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft update targets -----
        self.soft_update(self.actor, self.actor_target, self.config.tau)
        self.soft_update(self.critic, self.critic_target, self.config.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item())
        }
