import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env import QLearningEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ACTIONS = 101
OBS_DIM = 6

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999995
TARGET_UPDATE = 1_000
NUM_EPISODES = 200_000
EVAL_EVERY = 5_000
EVAL_EPISODES = 500
SAVE_PATH = "dqn_model.pt"


class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def obs_to_array(obs):
    return obs.astype(np.float32) / 100.0


def select_action(policy_net, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(policy_net(state_t).argmax(dim=1).item())


def train():
    env = QLearningEnv()
    policy_net = DQN(OBS_DIM, NUM_ACTIONS).to(DEVICE)
    target_net = DQN(OBS_DIM, NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)
    epsilon = EPSILON_START
    total_steps = 0
    reward_history = []

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = obs_to_array(obs)
        ep_reward = 0.0
        done = False

        while not done:
            action = select_action(policy_net, state, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_array(obs)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if len(buffer) >= BATCH_SIZE:
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)
                s_t = torch.tensor(s, device=DEVICE)
                a_t = torch.tensor(a, device=DEVICE).unsqueeze(1)
                r_t = torch.tensor(r, device=DEVICE)
                ns_t = torch.tensor(ns, device=DEVICE)
                d_t = torch.tensor(d, device=DEVICE)

                q_values = policy_net(s_t).gather(1, a_t).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(ns_t).max(dim=1).values
                    targets = r_t + GAMMA * next_q * (1.0 - d_t)

                loss = nn.functional.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        reward_history.append(ep_reward)

        if episode % EVAL_EVERY == 0:
            avg_train = np.mean(reward_history[-EVAL_EVERY:])
            eval_reward, eval_success = evaluate(policy_net, env)
            print(
                f"Episode {episode:>7,} | "
                f"Train avg: {avg_train:>9.1f} | "
                f"Eval avg: {eval_reward:>9.1f} | "
                f"Success: {eval_success:>5.1%} | "
                f"e: {epsilon:.4f} | "
                f"Steps: {total_steps:,}"
            )

    env.close()

    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")

    return policy_net


def evaluate(policy_net, env, n=EVAL_EPISODES):
    total = 0.0
    successes = 0

    for _ in range(n):
        obs, _ = env.reset()
        state = obs_to_array(obs)
        ep_reward = 0.0
        done = False

        while not done:
            action = select_action(policy_net, state, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs_to_array(obs)
            ep_reward += reward
            done = terminated or truncated

        total += ep_reward
        if info["battery"] >= 0:
            successes += 1

    return total / n, successes / n


def demo(policy_net, n=5):
    env = QLearningEnv(render_mode="human")

    for i in range(1, n + 1):
        print(f"\n{'='*50}")
        print(f"  Demo Episode {i}")
        print(f"{'='*50}")

        obs, _ = env.reset()
        state = obs_to_array(obs)
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            action = select_action(policy_net, state, epsilon=0.0)
            print(f"  Step {step}: Charging +{action}")

            obs, reward, terminated, truncated, info = env.step(action)
            state = obs_to_array(obs)
            total_reward += reward
            done = terminated or truncated
            step += 1

        if info["battery"] >= 0:
            print(f"\n  FINISHED! Battery: {info['battery']} | Total reward: {total_reward:.1f}")
        else:
            print(f"\n  STRANDED! Battery: {info['battery']} | Total reward: {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Training DQN agent for {NUM_EPISODES:,} episodes...")
    print()

    model = train()

    print("\nRunning demo episodes with learned policy...\n")
    demo(model)
