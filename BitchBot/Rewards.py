import copy
from turtle import speed
import numpy as n
from rlgym.utils import math
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self):
        self.orangeScore = 0
        self.blueScore = 0
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)

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
        # drive       = previous_action[0]  # Previous drive action
        # steer       = previous_action[1]  # Previous steer action
        # yaw         = previous_action[2]  # Previous yaw action
        # pitch       = previous_action[3]  # Previous pitch action
        # roll        = previous_action[4]  # Previous roll action
        # jump        = previous_action[5]  # Previous jump action
        # boost       = previous_action[6]  # Previous boost action
        # powerslide  = previous_action[7]  # Previous powerslide action
        # dist = n.sqrt(n.sum((carPos - ballPos) ** 2))  # Distance between car and ball

        ### START OF REWARDS ###

        # # Reward for being close to the ball
        # reward += n.exp(-dist / 20)

        # Reward for using few inputs
        # inputsAbsSum = n.sum(n.array(
        #     [
        #         # 0.01 * n.abs(drive),
        #         # 0.01 * n.abs(steer),
        #         # 0.01 * n.abs(yaw),
        #         # 0.01 * n.abs(pitch),
        #         n.abs(roll),
        #         jump,
        #         boost,
        #         powerslide
        #     ]
        # ))
        # reward -= inputsAbsSum / 400

        # Reward for facing the ball, (multiplied with carspeed)
        forward = player.car_data.forward()  # Vector in forward direction of car
        toBall = ballPos - carPos; toBall /= n.sqrt(n.dot(toBall, toBall))  # Normalized vetor from car to ball
        speedMultiplier = n.dot(carVel, forward) / 1410  # Car speed, normalized to 1 at max speed w/o boost
        facingMultiplier = n.dot(forward, toBall)
        rew_add = 0.02 * facingMultiplier * speedMultiplier

        if rew_add > 0:
             # Reduce reward for following the ball backwards
            if facingMultiplier < 0 and speedMultiplier < 0:
                rew_add *= 0.7

            reward += rew_add
        else:  # Punish driving away from ball more
            reward += 1.5 * rew_add
        

        # Reward for being on right side of the ball
        if ballPos[1] > carPos[1]:
            reward += 0.001
        else:
            reward -= 0.001

        # Reward for having the ball go fast toward opposition net
        ballToGoal = self.orangeGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
        ballVelScalar = n.linalg.norm(ballVel)
        if ballVelScalar > 0:
            ballVelNorm = ballVel / ballVelScalar
            reward += 0.02 * n.exp(-30 * n.abs(n.dot(ballVelNorm, ballToGoal) - 1)) * ballVelScalar / 1410

        # Punishment for having the ball go fast toward own net
        ballToGoal = self.blueGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
        # ballVelScalar = n.linalg.norm(ballVel)
        if ballVelScalar > 0:
            # ballVelNorm = ballVel / ballVelScalar
            reward -= 0.01 * n.exp(-30 * n.abs(n.dot(ballVelNorm, ballToGoal) - 1)) * ballVelScalar / 1410

        ballVelScalar = n.linalg.norm(ballVel)
        if player.ball_touched:
            # Reward for touching the ball, hard
            reward += 2 * ballVelScalar / 1410

            # Extra reward for touching the ball in the air
            if carPos[2] > 200:
                reward += 0.1 * carPos[2]

        # Update values to remember
        # self.prevCarPos = carPos
        # self.prevCarVel = carVel
        # self.prevBallPos = ballPos
        # self.prevBallVel = ballVel
        # self.prevDist = dist

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        reward = 0

        # Reward for scoring in the right goal
        if self.blueScore < state.blue_score:
            self.blueScore += 1
            reward += 10

        # Punishment for scoring in the wrong goal
        if self.orangeScore < state.orange_score:
            self.orangeScore += 1
            reward -= 10

        return reward