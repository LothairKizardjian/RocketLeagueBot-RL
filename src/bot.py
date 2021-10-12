import rlgym
import time
from src.simulator import BaseStateSetter, AerialStateSetter, BallTouchedTerminalCondition, CustomObsBuilder
from src.algorithms import RandomAgent
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 20

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

condition1 = TimeoutCondition(max_steps)
condition2 = BallTouchedTerminalCondition()

class SimpleBot():
    
    def run(self):
        #Make the default rlgym environment
        env = rlgym.make(state_setter=BaseStateSetter(),
                         terminal_conditions=[condition1, condition2],
                         reward_fn=VelocityPlayerToBallReward(),
                         obs_builder=CustomObsBuilder())
        
        agent = RandomAgent()
        
        while True:
            obs = env.reset()
            done = False
            steps = 0
            ep_reward = 0
            t0 = time.time()
            while not done:
                actions = agent.get_action(obs)
                new_obs, reward, done, state = env.step(actions)
                ep_reward += reward
                obs = new_obs
                steps += 1
            
        length = time.time() - t0
        print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))