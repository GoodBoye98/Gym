import numpy as n
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z

class BBStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        # randomCarPos = [n.random]
        
        # Random starting position and rotation
        x = n.random.rand() * 8000 - 4000
        y = n.random.rand() * 8000 - 4000
        z = 17
        yaw = n.random.rand() * 2 * n.pi - n.pi

        desired_car_pos = [x, y, z] #x, y, z
        desired_yaw = yaw
        
        # Loop over every car in the game.
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw
                
            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = [-1*coord for coord in desired_car_pos]
                yaw = -1*desired_yaw
                
            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of 
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
        # Now we will spawn the ball in the center of the field.
        x = n.random.rand() * 8000 - 4000
        y = n.random.rand() * 8000 - 4000
        state_wrapper.ball.set_pos(x=x, y=y, z=92.75)