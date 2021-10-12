### ABSTRACT CLASS FOR A BASE ALGORITHM ###

from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Any
import numpy as np

class BaseAgent(ABC):
    
    """
    
    Abstract class for a basic Agent.
    Describes the structure an Agent must implement.
    
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_action(self, observations, legal_actions) -> Any:
        """
        
        Parameters
        ----------
        observations : Observations of the game
        legal_actions : A list of legal actions
        
        List of actions :
            act[0] = throttle #[-1, 1] continuous
            act[1] = steer #[-1, 1] continuous
            act[2] = pitch #[-1, 1] continuous
            act[3] = yaw #[-1, 1] continuous
            act[4] = roll #[-1, 1] continuous
            act[5] = jump #{0, 1} discrete
            act[6] = boost #{0, 1} discrete
            act[7] = handbrake #{0, 1} discrete

        Returns : An action to apply to the environment

        """
        raise NotImplementedError
        
    @abstractmethod
    def update_policy(self, reward, action):
        raise NotImplementedError