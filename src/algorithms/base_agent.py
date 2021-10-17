### ABSTRACT CLASS FOR A BASE ALGORITHM ###

from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    
    """
    
    Abstract class for a basic Agent.
    Describes the structure an Agent must implement.
    
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def select_action(self, observations) -> Any:
        """
        
        Parameters
        ----------
        observations : Observations of the game
        
        Returns : An action to apply to the environment

        """
        raise NotImplementedError
        
    @abstractmethod
    def update_parameters(self):
        raise NotImplementedError
        
    @abstractmethod
    def train(self):
        raise NotImplementedError