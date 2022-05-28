import numpy as n
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z

class BBStateSetter(StateSetter):

    def _ballAroundCar(self, state_wrapper: StateWrapper) -> None:
        ## Parameters
        distMax = 1000
        distMin = 400
        heightMax = 92.75
        heighMin = 92.75

        while True:
            for car in state_wrapper.cars:
                # Random starting position and rotation
                x = n.random.uniform(-3500, 3500)
                y = n.random.uniform(-3500, 3500)
                yaw = n.random.uniform(-n.pi, n.pi)
                car.set_pos(x=x, y=y, z=17.01)
                car.set_rot(yaw=yaw)
            
            # Pick random car
            car = n.random.choice(state_wrapper.cars)
            x, y, z = car.position

            # Place ball randomly around the car
            rot = n.random.uniform(0, 2 * n.pi)
            dist = n.array([n.cos(rot), n.sin(rot)]) * (n.random.uniform(distMin, distMax))
            xBall = x + dist[0]
            yBall = y + dist[1]
            zBall = n.random.uniform(heighMin, heightMax)

            # Check if ball is within statium, otherwise try another random position
            if -3500 < xBall < 3500 and -3500 < yBall < 3500:
                state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)
                return

    def _ballInFrontOfCar(self, state_wrapper: StateWrapper) -> None:
        ## Parameters
        distMax = 1000
        distMin = 500
        square = 400
        heightMax = 92.75
        heighMin = 92.75

        while True:
            for car in state_wrapper.cars:
                # Random starting position and rotation
                x = n.random.uniform(-3500, 3500)
                y = n.random.uniform(-3500, 3500)
                yaw = n.random.uniform(-n.pi, n.pi)
                car.set_pos(x=x, y=y, z=17.01)
                car.set_rot(yaw=yaw)
            
            # Pick random car to have ball close
            car = n.random.choice(state_wrapper.cars)
            x, y, z = car.position

            # Place ball a little in front of the car, randomly in a 200x200 square
            dist = n.array([n.cos(yaw), n.sin(yaw)]) * (n.random.uniform(distMin, distMax))
            x_offset = n.random.uniform(-square/2, square/2)
            y_offset = n.random.uniform(-square/2, square/2)
            xBall = x + dist[0] + x_offset
            yBall = y + dist[1] + y_offset
            zBall = n.random.uniform(heighMin, heightMax)

            # Check if ball is within statium, otherwise try another random position
            if -3500 < xBall < 3500 and -3500 < yBall < 3500:
                state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)
                return

    def _ballInAir(self, state_wrapper: StateWrapper) -> None:
        ## Parameters
        velMax = 2000
        velMin = 200
        heightMax = 1900
        heighMin = 150

        # Set random car positions
        for car in state_wrapper.cars:
            # Random starting position and rotation
            x = n.random.uniform(-3500, 3500)
            y = n.random.uniform(-3500, 3500)
            yaw = n.random.uniform(-n.pi, n.pi)

            # Apply
            car.set_pos(x=x, y=y, z=17.01)
            car.set_rot(yaw=yaw)

        # Place ball in random location
        xBall = n.random.uniform(-3500, 3500)
        yBall = n.random.uniform(-3500, 3500)
        zBall = n.random.uniform(heighMin, heightMax)

        # Set random initial velocity to ball
        xVelBall = n.random.normal()
        yVelBall = n.random.normal()
        zVelBall = n.random.normal()

        # Normalize velocity to desiered range
        norm = (velMax - velMin) / n.sqrt(xVelBall**2 + yVelBall**2 + zVelBall**2) + velMin
        xVelBall *= norm
        yVelBall *= norm
        zVelBall *= norm

        # Apply position and velocity to ball
        state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)
        state_wrapper.ball.set_lin_vel(x=xVelBall, y=yVelBall, z=zVelBall)
        return

    def _ballOnCar(self, state_wrapper: StateWrapper) -> None:
        ## Parameters
        velMax = 2000
        velMin = 10

        for car in state_wrapper.cars:
            # Random starting position and rotation
            x = n.random.uniform(-3500, 3500)
            y = n.random.uniform(-3500, 3500)
            yaw = n.random.uniform(-n.pi, n.pi)

            cos = n.cos(yaw)
            sin = n.sin(yaw)

            # Random starting speed
            speed = n.random.uniform(0, 2000)
            xVel = cos * speed
            yVel = sin * speed

            # Set positions and velocities
            car.set_pos(x=x, y=y, z=17.01)
            car.set_lin_vel(x=xVel, y=xVel, z=0)
            car.set_rot(yaw=yaw)

        # Pick random car to have ball
        car = n.random.choice(state_wrapper.cars)
        x, y, z = car.position
        xVel, yVel, zVel = car.linear_velocity

        # Move ball on hood a little
        xAdd = cos * n.random.uniform(0, 20)
        yAdd = sin * n.random.uniform(0, 20)

        # Set ball on top of hood on random car
        state_wrapper.ball.set_pos(x=x+xAdd, y=y+yAdd, z=150)
        state_wrapper.ball.set_lin_vel(x=xVel, y=xVel, z=0)
        return

    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        # randomCarPos = [n.random]
        
        # Random starting position and rotation
        # x, y, z, yaw, xBall, yBall, zBall = self._ballAroundCar()
        # state_wrapper.ball.set_pos(x=xBall, y=yBall, z=zBall)
        
        # List of all state setters
        inits = [self._ballAroundCar, self._ballInAir, self._ballInFrontOfCar, self._ballOnCar]

        # Choose random state setter, and use it to set state
        init = n.random.choice(inits)
        init(state_wrapper)

        # Loop over every car in the game and set boost amount
        for car in state_wrapper.cars:
            car.boost = 0.33