import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import TRACKS

MAX_EXITS = max(len(t) for t in TRACKS)
OBS_DIM = 3 + MAX_EXITS * 3


class QLearningFullEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete(nvec=[101] * OBS_DIM)
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

        obs = np.zeros(OBS_DIM, dtype=np.int64)
        obs[0] = battery
        obs[1] = self._segment_idx
        obs[2] = total_exits

        for i, (d, tl, th) in enumerate(self._track):
            base = 3 + i * 3
            obs[base] = d
            obs[base + 1] = tl
            obs[base + 2] = th

        return obs

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
            battery, exit_num, total_exits = obs[0], obs[1], obs[2]
            print(f"  Battery: {battery:3d} | Exit {exit_num}/{total_exits}")
            for i in range(total_exits):
                base = 3 + i * 3
                marker = " <--" if i == exit_num else ""
                print(f"    Exit {i}: dist={obs[base]:>3}  traffic=[{obs[base+1]},{obs[base+2]}]{marker}")
        return None

    def close(self):
        pass
