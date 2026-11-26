"""Variable-action PPO actors and episodic centralized value baselines."""

import copy
import numpy as np
import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self, state_dim=40, action_dim=40):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def distribution(self, state, actions, mask):
        x = torch.cat([state.expand(len(actions), -1), actions], -1)
        logits = self.actor(x).squeeze(-1).masked_fill(~mask, -1e9)
        if not bool(mask.any()):
            raise ValueError("No legal actions")
        return torch.distributions.Categorical(logits=logits)

    def choose(self, state, actions, mask, deterministic=False):
        s = torch.as_tensor(state, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        m = torch.as_tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            d = self.distribution(s, a, m)
            ix = d.logits.argmax() if deterministic else d.sample()
            record = {
                "s": s,
                "a": a,
                "m": m,
                "ix": ix,
                "old_logp": d.log_prob(ix),
                "old_value": self.critic(s).squeeze(-1),
            }
        return int(ix), record


class PPO:
    def __init__(self, config):
        self.cfg = config
        self.agents = nn.ModuleList([Agent(), Agent()])
        self.optim = torch.optim.Adam(
            self.agents.parameters(), lr=config["learning_rate"]
        )
        self.buffer = []
        self.updates = 0

    def add_episode(self, records, rewards):
        # Separate returns/value baselines for the two players (competitive rewards have opposite signs).
        for player in range(2):
            ret = 0.0
            for record, r in zip(reversed(records[player]), reversed(rewards[player])):
                ret = float(r) + self.cfg["gamma"] * ret
                if record is not None:
                    self.buffer.append((player, record, ret))

    def update(self):
        if not self.buffer:
            return {}
        advantages = np.array(
            [ret - float(rec["old_value"]) for _, rec, ret in self.buffer],
            dtype=np.float32,
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        stats = []
        for _ in range(self.cfg["ppo_epochs"]):
            losses = []
            kls = []
            for (player, rec, ret), adv in zip(self.buffer, advantages):
                agent = self.agents[player]
                d = agent.distribution(rec["s"], rec["a"], rec["m"])
                lp = d.log_prob(rec["ix"])
                ratio = (lp - rec["old_logp"]).exp()
                a = torch.tensor(float(adv))
                clipped = (
                    ratio.clamp(1 - self.cfg["clip_ratio"], 1 + self.cfg["clip_ratio"])
                    * a
                )
                policy = -torch.minimum(ratio * a, clipped)
                value = (agent.critic(rec["s"]).squeeze(-1) - float(ret)).square()
                losses.append(policy + 0.5 * value - self.cfg["entropy"] * d.entropy())
                kls.append(rec["old_logp"] - lp.detach())
            loss = torch.stack(losses).mean()
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.agents.parameters(), self.cfg["max_grad_norm"]
            )
            self.optim.step()
            stats.append(
                {
                    "loss": float(loss.detach()),
                    "approx_kl": float(torch.stack(kls).mean()),
                    "transitions": len(losses),
                }
            )
        self.buffer = []
        self.updates += 1
        return stats[-1]
