import numpy as n
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BLUE_GOAL_CENTER, ORANGE_GOAL_CENTER

class BBStateSetter(StateSetter):

    def _ballOnSnoot(self, state_wrapper: StateWrapper) -> None:
        # Car to start with ball and 100 boost
        airdribbleCar = n.random.choice(state_wrapper.cars)
        airdribbleCar.boost = 100

        # Which goal to align toward slightly
        if airdribbleCar.team_num == BLUE_TEAM:
            goal = ORANGE_GOAL_CENTER
        else:
            goal = BLUE_GOAL_CENTER
        
        # Random position in the air
        posX = n.random.uniform(-2000, 2000)
        posY = n.random.uniform(-2000, 2000)
        posZ = n.random.uniform(650, 1800)

        # Ball to goal vector
        toGoal = goal - airdribbleCar.position; toGoal /= n.linalg.norm(toGoal)

        # Random rotation
        if airdribbleCar.team_num == BLUE_TEAM:
            rotPitch = n.random.uniform(0.8, 1.57)
            rotYaw   = n.random.uniform(-0.2, 0.2)
            rotRoll  = n.random.uniform(-3.14, 3.14)
        else:
            rotPitch = n.random.uniform(0.8, 1.57)
            rotYaw   = n.random.uniform(-0.2, 0.2) + n.pi
            if rotYaw > n.pi:
                rotYaw -= 2 * n.pi
            rotRoll  = n.random.uniform(-3.14, 3.14)

        # Random velocity, toward goal
        velX = toGoal[0] + n.random.uniform(-0.1, 0.1)
        velY = toGoal[1] + n.random.uniform(-0.1, 0.1)
        velZ = n.random.uniform(-0.1, 0.3)

        vel = n.array([velX, velY, velZ])
        vel *= n.random.uniform(250, 2000) / n.linalg.norm(vel)
        velX, velY, velZ = vel

        # Forward vector
        forward = n.array([
            n.cos(rotYaw)*n.cos(rotPitch),
            n.sin(rotYaw)*n.cos(rotPitch),
            n.sin(rotPitch)])

        # Place ball in front of car
        ballX = posX + forward[0] * (175 + n.random.uniform(-20, 20))
        ballY = posY + forward[1] * (175 + n.random.uniform(-20, 20))
        ballZ = posZ + forward[2] * (175 + n.random.uniform(-20, 20))

        # Set pos and vel of car and ball
        airdribbleCar.set_pos(x=posX, y=posY, z=posZ)
        airdribbleCar.set_lin_vel(x=velX, y=velY, z=velZ)
        airdribbleCar.set_rot(pitch=rotPitch, yaw=rotYaw, roll=rotRoll)

        state_wrapper.ball.set_pos(x=ballX, y=ballY, z=ballZ)
        state_wrapper.ball.set_lin_vel(x=velX, y=velY, z=velZ)

        # Set random pos and vel on ground for rest of cars
        for car in state_wrapper.cars:
            if car is airdribbleCar:
                continue

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
            car.boost = 0.33

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
            car.boost = 0.47

        # Pick random car to have ball
        car = n.random.choice(state_wrapper.cars)
        yaw = car.rotation[1]
        x, y, z = car.position
        xVel, yVel, zVel = car.linear_velocity

        # Move ball on hood a little
        xAdd = n.cos(yaw) * 15 + n.random.uniform(-20, 20)
        yAdd = n.sin(yaw) * 15 + n.random.uniform(-20, 20)

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
        # inits = [self._ballAroundCar, self._ballInAir, self._ballInFrontOfCar, self._ballOnCar]
        inits = [self._ballOnCar, self._ballOnSnoot]
        probs = [0.3, 0.7]

        # Choose random state setter, and use it to set state
        init = n.random.choice(inits, p=probs)
        init(state_wrapper)

        # # Loop over every car in the game and set boost amount
        # for car in state_wrapper.cars:
        #     car.boost = 0.33