import numpy as np

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward, TouchBallReward, LiuDistancePlayerToBallReward, FaceBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward  
from rlgym_tools.extra_rewards.diff_reward import DiffReward
    
class MyReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        reward = 0
        # reward += FaceBallReward().get_reward(player, state, previous_action)
        # reward += DiffReward(VelocityPlayerToBallReward()).get_reward(player, state, previous_action)
        reward += LiuDistancePlayerToBallReward().get_reward(player, state, previous_action)
        reward += 10*TouchBallReward().get_reward(player, state, previous_action)

        return reward