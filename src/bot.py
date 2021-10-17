import rlgym
import gym

from src.simulator import BaseStateSetter, AerialStateSetter, BallTouchedTerminalCondition, CustomObsBuilder, MyReward
from src.algorithms import PpoAgent
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition

class SimpleBot():
    
    def __init__(self, config, mode="train", retrain=1):
        self.config = config
        self.mode = mode
        self.retrain = retrain
        self.condition1 = NoTouchTimeoutCondition(self.config.max_steps)
        self.condition2 = BallTouchedTerminalCondition()
        self.conditions = [self.condition1]
    
    def run(self):        
        if self.config.env == "RocketLeague":
            env = rlgym.make(state_setter=BaseStateSetter(),
                         terminal_conditions=self.conditions,
                         reward_fn=MyReward(),
                         obs_builder=CustomObsBuilder())
        
        elif self.config.env == "LunarLander":
            env = gym.make("LunarLander-v2")        
            
        agent = PpoAgent(env, self.config)
        
        if self.mode == "train":
            if self.retrain != 1:
                agent.load()
            agent.train()
        elif self.mode == "test":
            agent.load()
            agent.test()
                