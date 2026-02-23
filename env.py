import gymnasium as gym
from gymnasium import spaces
import numpy as np


TRACKS = [
    [(5, 10, 15), (10, 5, 15), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(30, 10, 20), (40, 5, 10), (10, 15, 25), (40, 45, 60), (10, 5, 10), (5, 10, 20), (40, 5, 15)],
    [(5, 10, 15), (10, 5, 15), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 10, 15), (10, 5, 15), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 10, 15), (10, 5, 15), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 10, 15), (10, 5, 15), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(30, 10, 20), (40, 5, 10), (10, 15, 25), (40, 45, 60), (10, 5, 10), (5, 10, 20), (40, 5, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 5, 15), (10, 5, 20), (30, 15, 25), (15, 5, 20), (10, 10, 15)],
    [(5, 10, 25), (10, 10, 30), (30, 20, 30), (15, 5, 15), (10, 5, 10)],
]


class QLearningEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete(nvec=[101, 101, 101, 101, 101, 101])
        self.action_space = spaces.Discrete(101)
        self._track = None
        self._segment_idx = None
        self._battery = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._track = TRACKS[self.np_random.integers(len(TRACKS))]
        self._segment_idx = 0
        self._battery = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        charge_amount = int(action)
        self._battery = min(self._battery + charge_amount, 100)

        charge_penalty = -(30.0 + 0.2 * charge_amount ** 1.55) if charge_amount > 0 else 0.0

        dist, t_low, t_high = self._track[self._segment_idx]
        traffic = self.np_random.integers(t_low, t_high + 1)
        charge_used = dist + traffic
        travel_penalty = -float(charge_used)

        self._battery -= charge_used
        self._segment_idx += 1

        if self._battery < 0:
            reward = -200.0 + charge_penalty + travel_penalty
            terminated = True
            truncated = False
            if self.render_mode == "human":
                self._render_frame()
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if self._segment_idx >= len(self._track):
            reward = charge_penalty + travel_penalty
            terminated = True
            truncated = False
            if self.render_mode == "human":
                self._render_frame()
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        reward = charge_penalty + travel_penalty
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        battery = int(np.clip(self._battery, 0, 100))
        total_exits = len(self._track)

        if self._segment_idx < total_exits:
            dist, t_low, t_high = self._track[self._segment_idx]
        else:
            dist, t_low, t_high = 0, 0, 0

        return np.array(
            [battery, dist, t_low, t_high, self._segment_idx, total_exits],
            dtype=np.int64,
        )

    def _get_info(self):
        return {
            "battery": self._battery,
            "segment": self._segment_idx,
            "track_length": len(self._track) if self._track else 0,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        if self.render_mode == "human":
            obs = self._get_obs()
            print(
                f"  Battery: {obs[0]:3d} | "
                f"Next: dist={obs[1]} traffic=[{obs[2]},{obs[3]}] | "
                f"Exit {obs[4]}/{obs[5]}"
            )
        return None

    def close(self):
        pass
