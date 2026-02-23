"""Interactive terminal UI to play the EV charging game manually."""

import os
import sys
from env import QLearningEnv


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def battery_bar(level, max_val=100, width=30):
    filled = int(width * max(level, 0) / max_val)
    empty = width - filled

    if level > 50:
        color = "\033[92m"  # green
    elif level > 25:
        color = "\033[93m"  # yellow
    else:
        color = "\033[91m"  # red

    reset = "\033[0m"
    return f"{color}{'█' * filled}{'░' * empty}{reset} {level:>4}%"


def track_visual(track, segment_idx):
    parts = []
    for i, (dist, t_lo, t_hi) in enumerate(track):
        if i < segment_idx:
            parts.append(f"\033[90m ─({dist})─ ●\033[0m")  # dimmed, already passed
        elif i == segment_idx:
            parts.append(f"\033[96m ─({dist})─ ◆\033[0m")  # cyan, current
        else:
            parts.append(f" ─({dist})─ ○")
    car_pos = "🚗" if segment_idx < len(track) else ""
    finish = " 🏁" if segment_idx >= len(track) else ""
    return car_pos + "".join(parts) + finish


def render(obs, info, track, step_num, last_action=None, last_charge_used=None, reward=None):
    clear()
    battery, dist, t_low, t_high, exit_num, total_exits = obs

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               ⚡  EV CHARGING GAME  ⚡                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Step: {step_num:<5}                     Exit: {exit_num}/{total_exits}              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Battery: {battery_bar(battery)}              ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if last_action is not None:
        print(f"║  Last charge added: +{last_action:<39}║")
        if last_charge_used is not None:
            print(f"║  Charge consumed:   -{last_charge_used:<39}║")
        if reward is not None:
            print(f"║  Reward:            {reward:<40}║")
        print("╠══════════════════════════════════════════════════════════════╣")

    print("║  Track:                                                      ║")
    print(f"║  {track_visual(track, int(exit_num)):<69}║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if exit_num < total_exits:
        print("║  Next Segment:                                               ║")
        print(f"║    Distance:     {dist:<43}║")
        print(f"║    Traffic:      {t_low} - {t_high} (random){' ' * (33 - len(str(t_low)) - len(str(t_high)))}║")
        print(f"║    Worst case:   {dist + t_high} charge needed{' ' * (30 - len(str(dist + t_high)))}║")
        print("╠══════════════════════════════════════════════════════════════╣")

    print("╚══════════════════════════════════════════════════════════════╝")


def main():
    env = QLearningEnv()
    obs, info = env.reset()
    track = env._track

    step_num = 0
    last_action = None
    last_charge_used = None
    last_reward = None

    render(obs, info, track, step_num)

    while True:
        try:
            raw = input("\n  How much to charge? (0-100, q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if raw.lower() == "q":
            print("Goodbye!")
            break

        try:
            action = int(raw)
            if action < 0 or action > 100:
                print("  Enter a number between 0 and 100.")
                continue
        except ValueError:
            print("  Enter a number between 0 and 100.")
            continue

        battery_before = info["battery"]
        obs, reward, terminated, truncated, info = env.step(action)
        step_num += 1

        charged_to = min(battery_before + action, 100)
        last_charge_used = charged_to - info["battery"]
        last_action = action
        last_reward = reward

        render(obs, info, track, step_num, last_action, last_charge_used, last_reward)

        if terminated or truncated:
            if info["battery"] < 0:
                print(f"\n  \033[91m💀 STRANDED! Battery hit {info['battery']}. Game over.\033[0m")
            else:
                print(f"\n  \033[92m🎉 YOU MADE IT! Finished with {info['battery']} charge remaining.\033[0m")

            print(f"  Total reward: {last_reward}")

            try:
                again = input("\n  Play again? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if again == "y":
                obs, info = env.reset()
                track = env._track
                step_num = 0
                last_action = None
                last_charge_used = None
                last_reward = None
                render(obs, info, track, step_num)
            else:
                break

    env.close()


if __name__ == "__main__":
    main()
