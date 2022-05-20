import copy
import numpy as n
from rlgym.utils import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self):
        self.i = 0
        self.goals = 0

    def reset(self, initial_state: GameState):
        self.i = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        self.i += 1
        reward = 0

        # Car and ball positions
        carPos = player.car_data.position
        carVel = player.car_data.linear_velocity
        ballPos = state.ball.position
        ballVel = state.ball.linear_velocity

        # # Reward for not jumping all the fucking time
        # reward += 0.01 if player.on_ground else -0.02

        # Reward for driving facing the ball
        ang = math.quat_to_euler(player.car_data.quaternion)[1]
        forwardVec = n.array([n.cos(ang), n.sin(ang), 0])
        toBallVec = ballPos - carPos
        toBallVec /= n.sqrt(n.sum(toBallVec ** 2))
        carSpeed = n.sqrt(n.sum(carVel ** 2))
        reward += 1 / 15 * n.sum(forwardVec * toBallVec) * carSpeed / 2000

        # # Reward for ball rolling toward goal
        # rollVec = state.ball.linear_velocity
        # rollSquareMag = n.sum(rollVec ** 2)

        # if rollSquareMag > 0:
        #     toNetVec = n.array([0, 5200, 92]) - ballPos
        #     rollVec /= n.sqrt(rollSquareMag)
        #     toNetVec /= n.sqrt(n.sum(toNetVec ** 2))

        #     ballSpeed = n.sqrt(n.sum(ballVel ** 2))

        #     reward += 1 / 15 * n.sum(rollVec * toNetVec) * ballSpeed / 6000
        

        # Reward for scoring
        if self.goals < player.match_goals:
            self.goals += 1
            print('Scored!')
            reward += 10

        # Reward for touching the ball
        if player.ball_touched:
            reward += 1

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        return 0