import numpy as n
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z

class BBStateSetter(StateSetter):

    def _ballAroundCar(self, distMax=1000, distMin=400, heightMax=92.75, heighMin=92.75):
        while True:
            # Random starting position and rotation
            x = n.random.rand() * 7000 - 3500
            y = n.random.rand() * 7000 - 3500
            yaw = n.random.rand() * 2 * n.pi - n.pi

            # Place ball randomly around the car
            rot = n.random.rand() * 2 * n.pi
            dist = n.array([n.cos(rot), n.sin(rot)]) * (n.random.rand() * (distMax - distMin) + distMin)
            xBall = x + dist[0]
            yBall = y + dist[1]
            zBall = n.random.rand() * (heightMax - heighMin) + heighMin

            # Check if ball is within statium, otherwise try another random position
            if -3500 < xBall < 3500 and -3500 < yBall < 3500:
                return x, y, 18.33, yaw, xBall, yBall, 92.75

    def _ballInFrontOfCar(self, distMax=1000, distMin=500, square=400, heightMax=92.75, heighMin=92.75):
        while True:
            # Random starting position and rotation
            x = n.random.rand() * 7000 - 3500
            y = n.random.rand() * 7000 - 3500
            yaw = n.random.rand() * 2 * n.pi - n.pi

            # Place ball a little in front of the car, randomly in a 200x200 square
            dist = n.array([n.cos(yaw), n.sin(yaw)]) * (n.random.rand() * (distMax - distMin) + distMin)
            x_offset = n.random.rand() * square - square / 2
            y_offset = n.random.rand() * square - square / 2
            xBall = x + dist[0] + x_offset
            yBall = y + dist[1] + y_offset
            zBall = n.random.rand() * (heightMax - heighMin) + heighMin

            # Check if ball is within statium, otherwise try another random position
            if -3500 < xBall < 3500 and -3500 < yBall < 3500:
                return x, y, 18.33, yaw, xBall, yBall, 92.75

    def _ballInAir(self, velMax=2000, velMin=200, heightMax=1900, heighMin=150):
        # Random starting position and rotation
        x = n.random.rand() * 7000 - 3500
        y = n.random.rand() * 7000 - 3500
        yaw = n.random.rand() * 2 * n.pi - n.pi

        # Place ball in random location
        xBall = n.random.rand() * 7000 - 3500
        yBall = n.random.rand() * 7000 - 3500
        zBall = n.random.rand() * (heightMax - heighMin) + heighMin

        # Set random initial velocity to ball
        xVelBall = n.random.normal()
        yVelBall = n.random.normal()
        zVelBall = n.random.normal()

        # Normalize velocity to desiered range
        norm = (velMax - velMin) / n.sqrt(xVelBall**2 + yVelBall**2 + zVelBall**2) + velMin
        xVelBall *= norm
        yVelBall *= norm
        zVelBall *= norm

        return x, y, 18.33, yaw, xBall, yBall, zBall, xVelBall, yVelBall, zVelBall 


    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        # randomCarPos = [n.random]
        
        # Random starting position and rotation
        # x, y, z, yaw, xBall, yBall, zBall = self._ballAroundCar()
        # state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)

        x, y, z, yaw, xBall, yBall, zBall, xVelBall, yVelBall, zVelBall = self._ballInAir()
        state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)
        state_wrapper.ball.set_lin_vel(x=xVelBall, y=yVelBall, z=zVelBall)      

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