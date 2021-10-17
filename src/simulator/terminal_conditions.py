from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState

class BallTouchedTerminalCondition(TerminalCondition):
    # Stops the episode if the ball is touched
    
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
      return current_state.last_touch != -1