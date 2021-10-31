import os
import sys
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")

from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    def __init__(self, width: int = 20, height: int = 20, max_rooms: int = 3,
                 min_room_xy: int = 5, max_room_xy: int = 12,
                 observation_size: int = 11, vision_radius: int = 5, max_steps: int = 2000):
        super().__init__(width=width, height=height, max_rooms=max_rooms,
                         min_room_xy=min_room_xy, max_room_xy=max_room_xy,
                         observation_size=observation_size,
                         vision_radius=vision_radius, max_steps=max_steps)
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3])
        self.action_space = spaces.Discrete(3)

    def reset(self) -> np.ndarray:
        observation = super().reset()
        return observation[:, :, :-1]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, done, info = super().step(action)
        observation = observation[:, :, :-1]  # remove trajectory
        return observation, reward, done, info
