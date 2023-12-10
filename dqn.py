import compiler_gym
from compiler_gym import wrappers
import collections
import random
import numpy as np
import sys, os
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
buffer_limit = 500_000
batch_size = 32

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device:{device}")


class ReplayBuffer:
    def __init__(self, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float32).to(self.device),
            torch.tensor(a_lst, dtype=torch.int64).to(self.device),
            torch.tensor(np.array(r_lst), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(done_mask_lst), dtype=torch.int64).to(self.device),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, n_input_channels, n_actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(n_input_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.tensor(obs).unsqueeze(0).to(device).to(dtype=torch.float32)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 123)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    Q = q(s).gather(1, a)
    Q_prime_vals = q(s_prime)
    argmax_Q_a = Q_prime_vals.max(1)[1]

    # Rt+1 + γ max_a Q(S_t+1, a; θt). where θ=θ- because we update target params to train params every t steps
    Q_next = q_target(s_prime)
    q_target_s_a_prime = Q_next.gather(1, argmax_Q_a.unsqueeze(1))
    y_i = r + done_mask * gamma * q_target_s_a_prime

    assert Q.shape == y_i.shape, f"{Q.shape}, {y_i.shape}"
    loss = F.smooth_l1_loss(Q, y_i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return Q.mean()


def main():
    env = compiler_gym.make(
        "llvm-autophase-ic-v0",
        benchmark="benchmark://cbench-v1/sha",
        observation_space="Autophase",
    )
    env = wrappers.TimeLimit(env, 45)
    env.reset()
    env.write_bitcode("sha.bc")

    # set seed for reproducibility
    seed = int(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    q = Qnet(56, 124)
    q_target = Qnet(56, 124)
    q_target.load_state_dict(q.state_dict())
    q.to(device)
    q_target.to(device)
    memory = ReplayBuffer(device)

    total_frames = 200  # Total number of frames for annealing
    print_interval = 1
    train_update_interval = 4
    target_update_interval = 500
    train_start = 1_000
    score = 0
    best_score = 0
    step = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    q_value = torch.tensor(0)

    with open(
        f"data/llvm-autophase/Uniform-DDQN-sha{sys.argv[1]}.csv", "w", newline=""
    ) as csvfile:
        fieldnames = ["episode", "step", "score", "Q-value", "state", "action-sequence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n_epi in range(4_000):
            observation = env.reset()
            terminated = False
            patience = 5
            change_count = 0
            episode_length = 124
            actions_taken = 0
            action_sequence = []
            while not terminated and actions_taken < episode_length:
                epsilon = max(
                    0.1, 1.0 - 0.01 * (step / total_frames)
                )  # Linear annealing from 1.0 to 0.1
                action = q.sample_action(observation, epsilon)
                action_sequence.append(int(action))
                observation_prime, reward, terminated, info = env.step(action)
                actions_taken += 1
                if reward == 0:
                    change_count += 1
                else:
                    change_count = 0
                done_mask = (
                    0.0 if (terminated or actions_taken < episode_length) else 1.0
                )

                memory.put((observation, action, reward, observation_prime, done_mask))
                observation = observation_prime

                if step > train_start and step % train_update_interval == 0:
                    q_value = train(q, q_target, memory, optimizer)

                if step > train_start and step % target_update_interval == 0:
                    q_target.load_state_dict(q.state_dict())

                score += reward
                step += 1
                if terminated or change_count > patience:
                    break

            if n_epi % print_interval == 0:
                print(
                    f"episode :{n_epi}, step: {step}, score : {score/print_interval:.2f}, n_buffer : {memory.size()}, eps : {epsilon*100:.1f}%"
                )
                writer.writerow(
                    {
                        "episode": n_epi,
                        "step": step,
                        "score": score / print_interval,
                        "Q-value": q_value.item(),
                        "state": observation,
                        "action-sequence": action_sequence,
                    }
                )
                csvfile.flush()
                if score > best_score:
                    torch.save(q.state_dict(), "models/sha_model")
                score = 0.0
        env.write_bitcode("sha-optimized.bc")
    env.close()


if __name__ == "__main__":
    main()
