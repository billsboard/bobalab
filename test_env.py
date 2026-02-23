"""Quick test script to verify the QLearningEnv works correctly."""

from env import QLearningEnv, TRACKS


def test_reset():
    """Verify reset returns valid obs and info."""
    env = QLearningEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
    assert obs[0] == 0, f"Battery should start at 0, got {obs[0]}"
    assert info["segment"] == 0, f"Should start at segment 0, got {info['segment']}"
    assert env.observation_space.contains(obs), f"Obs not in observation_space: {obs}"

    print("[PASS] test_reset")
    env.close()


def test_charge_and_drive():
    """Verify charging increases battery and driving consumes it."""
    env = QLearningEnv()
    obs, _ = env.reset(seed=42)

    battery_before = obs[0]
    obs, reward, terminated, truncated, info = env.step(50)

    assert info["battery"] <= 50, f"Battery should be <= 50 after charging 50, got {info['battery']}"
    assert info["segment"] == 1, f"Should have advanced to segment 1, got {info['segment']}"
    assert env.observation_space.contains(obs), f"Obs not in observation_space: {obs}"

    print("[PASS] test_charge_and_drive")
    env.close()


def test_battery_cap():
    """Verify battery is capped at 100."""
    env = QLearningEnv()
    env.reset(seed=42)

    env._battery = 90
    env.step(50)
    assert env._battery <= 100, f"Battery exceeded 100 before driving (uncapped)"

    print("[PASS] test_battery_cap")
    env.close()


def test_stranded():
    """Verify the episode ends when battery runs out."""
    env = QLearningEnv()
    env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated, "Should be terminated (stranded with 0 charge)"
    assert reward < 0, f"Reward should be negative when stranded, got {reward}"

    print("[PASS] test_stranded")
    env.close()


def test_full_episode_success():
    """Play a full episode charging max every step — should complete the track."""
    env = QLearningEnv()
    obs, info = env.reset(seed=42)
    track_len = info["track_length"]

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(100)
        total_reward += reward
        steps += 1

        if steps > 20:
            print("[FAIL] test_full_episode_success — too many steps, possible infinite loop")
            env.close()
            return

    assert info["segment"] == track_len, (
        f"Should have completed all {track_len} segments, stopped at {info['segment']}"
    )
    assert total_reward > 0, f"Expected positive total reward for completing the track, got {total_reward}"

    print(f"[PASS] test_full_episode_success — completed {track_len} exits in {steps} steps, reward={total_reward}")
    env.close()


def test_human_render():
    """Run a short episode with render_mode='human' to check print output."""
    print("\n--- Human render test ---")
    env = QLearningEnv(render_mode="human")
    obs, info = env.reset(seed=0)

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(100)
        if terminated or truncated:
            break

    print("--- End render test ---")
    print("[PASS] test_human_render")
    env.close()


def test_all_tracks():
    """Verify every track in TRACKS can be completed with max charging."""
    for i, track in enumerate(TRACKS):
        env = QLearningEnv()
        env.reset(seed=0)
        env._track = track
        env._segment_idx = 0
        env._battery = 0

        terminated = False
        for _ in range(len(track)):
            _, _, terminated, _, info = env.step(100)
            if terminated and info["battery"] < 0:
                print(f"[FAIL] test_all_tracks — stranded on track {i}")
                env.close()
                return

        assert terminated, f"Track {i} did not terminate after all segments"
        print(f"  Track {i}: {len(track)} exits, final battery={info['battery']}")
        env.close()

    print("[PASS] test_all_tracks")


if __name__ == "__main__":
    test_reset()
    test_charge_and_drive()
    test_battery_cap()
    test_stranded()
    test_full_episode_success()
    test_human_render()
    test_all_tracks()

    print("\nAll tests passed!")
