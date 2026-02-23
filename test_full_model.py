import numpy as np
import torch
from env import TRACKS
from env_full import QLearningFullEnv, OBS_DIM
from train_full import DQN, NUM_ACTIONS, obs_to_array, SAVE_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(OBS_DIM, NUM_ACTIONS).to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def best_action(state):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(model(state_t).argmax(dim=1).item())


def battery_bar(level, width=30):
    level = max(level, 0)
    filled = int(width * level / 100)
    empty = width - filled
    if level > 50:
        color = "\033[92m"
    elif level > 25:
        color = "\033[93m"
    else:
        color = "\033[91m"
    return f"{color}{'█' * filled}{'░' * empty}\033[0m {level:>4}"


def track_map(track, seg_idx):
    parts = []
    for i, (d, tl, th) in enumerate(track):
        if i < seg_idx:
            parts.append(f"\033[90m--({d})--*\033[0m")
        elif i == seg_idx:
            parts.append(f"\033[96m--({d})->[CAR]\033[0m")
        else:
            parts.append(f"--({d})--o")
    flag = " [FINISH]" if seg_idx >= len(track) else ""
    return "".join(parts) + flag


def run_track(track, track_name, runs=10):
    print(f"\n{'=' * 70}")
    print(f"  {track_name}")
    print(f"  Segments: {len(track)}  |  "
          f"Total min distance: {sum(d + tl for d, tl, _ in track)}  |  "
          f"Total max distance: {sum(d + th for d, _, th in track)}")
    print(f"{'=' * 70}")

    for seg_i, (d, tl, th) in enumerate(track):
        print(f"  Exit {seg_i}: dist={d:>3}  traffic=[{tl}, {th}]  worst={d + th}")
    print(f"{'─' * 70}")

    all_rewards = []
    all_successes = []

    for run in range(1, runs + 1):
        env = QLearningFullEnv()
        obs, _ = env.reset()
        env._track = track
        env._segment_idx = 0
        env._battery = 0
        obs = env._get_obs()

        state = obs_to_array(obs)
        total_reward = 0.0
        done = False
        steps = []

        while not done:
            action = best_action(state)
            battery_before = env._battery

            obs, reward, terminated, truncated, info = env.step(action)
            state = obs_to_array(obs)
            total_reward += reward
            done = terminated or truncated

            steps.append({
                "charge": action,
                "battery_after_charge": min(battery_before + action, 100),
                "battery_after_drive": info["battery"],
                "reward": reward,
            })

        success = info["battery"] >= 0
        all_rewards.append(total_reward)
        all_successes.append(success)

        print(f"\n  Run {run}")
        print(f"  {track_map(track, info['segment'])}")
        print()

        header = f"  {'Exit':>4} | {'Charged':>7} | {'Battery->':>9} | {'After Drive':>11} | {'Reward':>10}"
        print(header)
        print(f"  {'─' * 4}─┼─{'─' * 7}─┼─{'─' * 9}─┼─{'─' * 11}─┼─{'─' * 10}")

        for i, s in enumerate(steps):
            batt_bar_mini = battery_bar(max(s["battery_after_drive"], 0), width=10)
            print(
                f"  {i:>4} | +{s['charge']:>5} | {s['battery_after_charge']:>8} | "
                f"{s['battery_after_drive']:>7}  {batt_bar_mini} | {s['reward']:>10.1f}"
            )

        status = "\033[92mFINISHED\033[0m" if success else "\033[91mSTRANDED\033[0m"
        print(f"\n  {status}  |  Final battery: {info['battery']}  |  Total reward: {total_reward:.1f}")

        env.close()

    avg_reward = np.mean(all_rewards)
    success_rate = sum(all_successes) / len(all_successes)
    print(f"\n  {'─' * 50}")
    print(f"  Summary over {runs} runs:")
    print(f"    Avg reward:   {avg_reward:.1f}")
    print(f"    Success rate: {success_rate:.0%}")
    print(f"  {'─' * 50}")


if __name__ == "__main__":
    print("\033[1m")
    print("  EV Charging DQN (Full Obs) -- Model Test")
    print(f"  Device: {DEVICE}")
    print("\033[0m")

    unique_tracks = {
        "Track A - Main Short (5 exits, Rounds 1-6 original)": TRACKS[0],
        "Track B - Main Long (7 exits, Rounds 2,7 original)": TRACKS[1],
        "Track C - Short 1 (5 exits, Rounds 1-6 new)": TRACKS[7],
        "Track D - Short 2 (5 exits, Round 7 new)": TRACKS[13],
    }

    custom_tracks = {
        "Track E (unseen -- short easy)": [
            (8, 2, 5),
            (12, 3, 6),
            (10, 4, 8),
        ],
        "Track F (unseen -- long hard)": [
            (25, 10, 20),
            (30, 15, 25),
            (20, 5, 15),
            (35, 20, 30),
            (15, 5, 10),
            (40, 10, 20),
        ],
    }

    for name, track in unique_tracks.items():
        run_track(track, name)

    print(f"\n{'=' * 70}")
    print("  Testing on UNSEEN tracks (not in training data)")
    print(f"{'=' * 70}")

    for name, track in custom_tracks.items():
        run_track(track, name)
