import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


if torch.cuda.is_available():
    GPU_num = torch.cuda.current_device()
    device = torch.device("cuda:{}".format(GPU_num))
else:
    device = "cpu"


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_numb, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.pi = nn.Linear(hidden_size, action_numb)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.pi(x)
        return F.softmax(x, dim=-1)


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_numb, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_numb)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AdversaryNet(nn.Module):

    def __init__(self, state_dim, hidden_size=128, alpha=0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, state_dim)
        self.alpha = alpha

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.alpha * torch.tanh(x)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, action_numb, size=100000):
        self.obs = np.zeros([size, obs_dim], dtype=np.float32)
        self.next = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts = np.zeros([size, act_dim], dtype=np.int64)
        self.rews = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.max = size

    def add(self, o, a, r, n, d):
        self.obs[self.ptr] = o
        self.next[self.ptr] = n
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch=64):
        idx = np.random.randint(0, self.size, size=batch)
        return (torch.tensor(self.obs[idx]).to(device),
                torch.tensor(self.acts[idx]).to(device),
                torch.tensor(self.rews[idx]).to(device),
                torch.tensor(self.next[idx]).to(device),
                torch.tensor(self.done[idx]).to(device))

class Agent:
    def __init__(self, state_dim, action_dim, action_numb):
        self.actor = ActorNet(state_dim, action_numb).to(device)
        self.qf1 = CriticNet(state_dim, action_numb).to(device)
        self.qf2 = CriticNet(state_dim, action_numb).to(device)
        self.qf1_t = CriticNet(state_dim, action_numb).to(device)
        self.qf2_t = CriticNet(state_dim, action_numb).to(device)
        self.qf1_t.load_state_dict(self.qf1.state_dict())
        self.qf2_t.load_state_dict(self.qf2.state_dict())

        self.adversary = AdversaryNet(state_dim).to(device)

        self.buffer = ReplayBuffer(state_dim, action_dim, action_numb)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.q1_opt = optim.Adam(self.qf1.parameters(), lr=1e-3)
        self.q2_opt = optim.Adam(self.qf2.parameters(), lr=1e-3)
        self.adversary_opt = optim.Adam(self.adversary.parameters(), lr=1e-4)

        self.dual_cst = torch.ones(1, requires_grad=True, device=device)
        self.dual_opt = optim.Adam([self.dual_cst], lr=5e-4)

        self.gamma = 0.95
        self.target_robust = 0.0001
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_numb = action_numb

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        js = 0.5 * torch.sum(p * torch.log(p / (m + 1e-8) + 1e-8), dim=1) + \
             0.5 * torch.sum(q * torch.log(q / (m + 1e-8) + 1e-8), dim=1)
        return js

    def generate_optimal_perturbation(self, state):
        if isinstance(state, np.ndarray):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if len(state.shape) == 1 else state

        with torch.no_grad():
            perturbation = self.adversary(state_tensor)
            perturbed_state = state_tensor + perturbation

        return perturbed_state.squeeze(), perturbation.squeeze()

    def apply_adversarial_perturbation(self, state, reference_state=None):
        perturbed_state, _ = self.generate_optimal_perturbation(state)

        if isinstance(perturbed_state, torch.Tensor):
            return perturbed_state.cpu().numpy()
        return perturbed_state

    def apply_adversarial_perturbation_batch(self, states):
        batch_size = states.shape[0]
        perturbed_states = states.clone()
        perturbation_info = []

        for i in range(batch_size):
            state = states[i:i + 1]

            perturbation = self.adversary(state)
            perturbed_state = state + perturbation
            perturbed_states[i] = perturbed_state.squeeze()

            original_prob = self.actor(state)
            perturbed_prob = self.actor(perturbed_state)
            js_val = self.js_divergence(original_prob, perturbed_prob)
            perturbation_info.append(js_val.item())

        return perturbed_states, torch.tensor(perturbation_info, device=device)

    def policy_expand(self, state):

        # perturbed_state = self.apply_adversarial_perturbation(state)
        act_probs = self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        act_probs = zip(list(range(self.action_dim)), act_probs.data.cpu().numpy().flatten())

        return act_probs

    def select_action_batch(self, state):
        prob = self.actor(state)
        m = Categorical(prob)
        a = m.sample().unsqueeze(1)
        return a, prob

    def train(self, mode: bool = True):
        self.actor.train(mode)
        self.qf1.train(mode)
        self.qf2.train(mode)
        self.adversary.train(mode)
        return self

    def train_model(self, batch_size=64):
        torch.autograd.set_detect_anomaly(False)

        if self.buffer.size < batch_size:
            return

        obs, acts, rews, nexts, dn = self.buffer.sample(batch_size)
        obs_adv = obs.detach()
        nexts_adv = nexts.detach()

        obs_perturbation_adv = self.adversary(obs_adv)
        nexts_perturbation_adv = self.adversary(nexts_adv)
        perturbed_obs_adv = obs_adv + obs_perturbation_adv
        perturbed_nexts_adv = nexts_adv + nexts_perturbation_adv

        original_obs_prob_adv = self.actor(obs_adv).detach()
        perturbed_obs_prob_adv = self.actor(perturbed_obs_adv)
        original_nexts_prob_adv = self.actor(nexts_adv).detach()
        perturbed_nexts_prob_adv = self.actor(perturbed_nexts_adv)

        js_obs_adv = self.js_divergence(original_obs_prob_adv, perturbed_obs_prob_adv)
        js_nexts_adv = self.js_divergence(original_nexts_prob_adv, perturbed_nexts_prob_adv)
        js_constraint_adv = (js_obs_adv + js_nexts_adv) / 2.0
        adversary_loss = -js_constraint_adv.mean()

        self.adversary_opt.zero_grad()
        adversary_loss.backward()
        self.adversary_opt.step()

        obs_perturbation = self.adversary(obs)
        nexts_perturbation = self.adversary(nexts)

        perturbed_obs = obs + obs_perturbation
        perturbed_nexts = nexts + nexts_perturbation

        original_obs_prob = self.actor(obs)
        perturbed_obs_prob = self.actor(perturbed_obs)
        original_nexts_prob = self.actor(nexts)
        perturbed_nexts_prob = self.actor(perturbed_nexts)

        js_obs = self.js_divergence(original_obs_prob, perturbed_obs_prob)
        js_nexts = self.js_divergence(original_nexts_prob, perturbed_nexts_prob)
        js_constraint = (js_obs + js_nexts) / 2.0

        a, p = self.select_action_batch(perturbed_obs)
        a2, p2 = self.select_action_batch(perturbed_nexts)

        q1 = self.qf1(obs).gather(1, acts.long()).squeeze(1)
        q2 = self.qf2(obs).gather(1, acts.long()).squeeze(1)

        with torch.no_grad():
            q1n = self.qf1_t(perturbed_nexts)
            q2n = self.qf2_t(perturbed_nexts)
            minq = torch.min(q1n, q2n)

        v_b = (p2 * minq).sum(-1) - self.dual_cst.exp() * js_constraint
        q_b = rews + self.gamma * (1.0 - dn) * v_b

        with torch.no_grad():
            q1_pi = self.qf1(perturbed_obs)
            q2_pi = self.qf2(perturbed_obs)
            min_q = torch.min(q1_pi, q2_pi)

        a_loss = (self.dual_cst.exp() * js_constraint - (p * min_q).sum(-1)).mean()

        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        l1 = F.mse_loss(q1, q_b.detach())
        l2 = F.mse_loss(q2, q_b.detach())
        self.q1_opt.zero_grad()
        l1.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        l2.backward()

        cst_loss = (self.dual_cst.exp() * (self.target_robust - js_constraint.detach())).mean()
        self.dual_opt.zero_grad()
        cst_loss.backward()
        self.dual_opt.step()

        with torch.no_grad():
            for p, t in zip(self.qf1.parameters(), self.qf1_t.parameters()):
                t.mul_(0.995).add_(0.005 * p)
            for p, t in zip(self.qf2.parameters(), self.qf2_t.parameters()):
                t.mul_(0.995).add_(0.005 * p)

    def save_model(self, model_name, model_path):
        name = './' + model_path + '/actor%d' % model_name
        torch.save(self.actor, "{}.pkl".format(name))
        print("The model is saved!!!")

