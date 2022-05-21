import copy
import numpy as n
from rlgym.utils import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self):
        self.goals = 0

    def reset(self, initial_state: GameState):
        self.prevCarPos = 0
        self.prevCarVel = 0
        self.prevBallPos = 0
        self.prevBallVel = 0
        self.prevDist = 0


    def get_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        ### Order of actions in previous_action
        ### [drive, steer, yaw, pitch, roll, jump, boost, powerslide] ###
        
        # Initialize reward
        reward = 0

        # Car and ball positions
        carPos = player.car_data.position
        carVel = player.car_data.linear_velocity
        ballPos = state.ball.position
        ballVel = state.ball.linear_velocity

        # Helpful variables
        drive       = previous_action[0]  # Previous drive action
        steer       = previous_action[1]  # Previous steer action
        yaw         = previous_action[2]  # Previous yaw action
        pitch       = previous_action[3]  # Previous pitch action
        roll        = previous_action[4]  # Previous roll action
        jump        = previous_action[5]  # Previous jump action
        boost       = previous_action[6]  # Previous boost action
        powerslide  = previous_action[7]  # Previous powerslide action
        dist = n.sqrt(n.sum((carPos - ballPos) ** 2))  # Distance between car and ball

        ### START OF REWARDS ###

        # # Reward for being close to the ball
        # reward += n.exp(-dist / 20)

        # Reward using few inputs
        inputsSquareSum = n.sum(n.array([drive, steer, yaw, pitch, roll, jump, boost, powerslide]) ** 2)
        reward -= inputsSquareSum / 100

        # # Reward for moving closer to the ball
        # if self.prevDist:
        #     reward += (dist - self.prevDist) / (1410 * 300)

        # Reward for facing the ball, (multiplied with carspeed)
        yaw = player.car_data.yaw()
        forward = n.array([n.cos(yaw), n.sin(yaw), 0])  # Vector in forward direction of car
        toBall = ballPos - carPos; toBall /= n.sqrt(n.sum(toBall**2))  # Normalized vetor from car to ball
        speedMultiplier = n.sqrt(n.sum(carVel**2)) / 1410  # Car speed, normalized to 1 at max speed w/o boost
        reward += 1/15 * n.sum(forward * toBall)  # cos of angle between the two vectors


        # # Reward for touching the ball
        # if player.ball_touched:
        #     reward += 1.0

        # # Reward for scoring
        # if self.goals < player.match_goals:
        #     self.goals += 1
        #     print('Scored!')
        #     reward += 10

        # Update values to remember
        self.prevCarPos = carPos
        self.prevCarVel = carVel
        self.prevBallPos = ballPos
        self.prevBallVel = ballVel
        self.prevDist = dist

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        return 0