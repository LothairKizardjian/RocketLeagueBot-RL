from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np


class BaseStateSetter(StateSetter):
    # Sets the ball like it would be at the start of a match. Usefull for training kickoffs, getting to the ball etc ...
    
    def reset(self, state_wrapper : StateWrapper):
        # set pos and rotation of the car
        desired_car_pos = [0, -2100, 17]
        desired_yaw = np.pi/2
        
        for car in state_wrapper.cars:
            car.set_pos(*desired_car_pos)
            car.set_rot(yaw=desired_yaw)
            car.boost = 0.33
        
        # spawing the ball mid air
        state_wrapper.ball.set_pos(x=0, y=0, z=17)

class AerialStateSetter(StateSetter):
    # Sets a ball in the air. Usefull for training aerial car control, aerial mechanics etc ...
    
    def reset(self, state_wrapper : StateWrapper):
        # set pos and rotation of the car
        desired_car_pos = [0, -2100, 17]
        desired_yaw = np.pi/2
        
        for car in state_wrapper.cars:
            car.set_pos(*desired_car_pos)
            car.set_rot(yaw=desired_yaw)
            car.boost = 0.33
        
        # spawing the ball mid air
        state_wrapper.ball.set_pos(x=0, y=0, z=CEILING_Z - 100)