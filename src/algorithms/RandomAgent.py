from src.algorithms.BaseAgent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    
    def __init__(self):
        super().__init__()
        self.legal_actions = np.array([np.ones(8, dtype=np.float32),-1*np.ones(8, dtype=np.float32)])
        self.legal_actions[1,5:] = 0
        self.policy = np.zeros(8, dtype=np.float32)
        self.policy += 1./8.
        
    def get_action(self, observations):
        x = np.random.randint(2)
        y = np.random.choice(np.arange(8), p=self.policy)
        action = self.legal_actions[x,y]
        actions = np.zeros(8, dtype=np.float32)
        actions[y] += action
        return actions
        
    def update_policy(self,reward,action):
        pass
        