from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
import numpy as np

class CustomObsBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        obs = []
        obs.append(state.ball.serialize())
        for player in state.players:
            obs.append(player.car_data.serialize())
        obs.append(state.boost_pads)
        return obs