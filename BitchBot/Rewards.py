import copy
from turtle import speed
import numpy as n
from rlgym.utils import math
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, SUPERSONIC_THRESHOLD, CEILING_Z, ORANGE_TEAM, BLUE_TEAM
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self,
        ballTouchMultiplier         = 1.1,      # 1.3
        ballAccelerateMultiplier    = 0.5,      # 0.7
        shotOnGoalMultiplier        = 1.7,      # 2.0
        positionMultiplier          = 0.05,     # 0.05
        speedMultiplier             = 0.9,      # 1.2
        heightMultiplier            = 0.9,      # 1.2
        faceBallMultiplier          = 0.2,      # 0.1
        ownGoalReward               = -2.0,     # -2.0
        goalReward                  = 2.0,      # 2.0
        yVelocityReward             = 0.3,      # 0.3
    ):
        self.orangeScore = 0
        self.blueScore = 0
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)

        ###  REWARD MULTIPLIERS
        self.ballTouchMultiplier = ballTouchMultiplier                  # r per sec touching ball
        self.ballAccelerateMultiplier = ballAccelerateMultiplier        # r per 0->supersonic
        self.shotOnGoalMultiplier = shotOnGoalMultiplier                # r shot straight at net at supersonic
        self.positionMultiplier = positionMultiplier                    # r per sec in right position
        self.speedMultiplier = speedMultiplier                          # r per sec at supersonic
        self.heightMultiplier = heightMultiplier                        # r per sec at supersonic
        self.faceBallMultiplier = faceBallMultiplier                    # r per sec at supersonic
        self.ownGoalReward = ownGoalReward                              # r per own goal
        self.goalReward = goalReward                                    # r per goal
        self.yVelocityReward = yVelocityReward                          # r per positive supersonic velocity change in
        ###  REWARD MULTIPLIERS

    def reset(self, initial_state: GameState):
        self.prevCarPos = n.array([0, 0, 0], dtype=n.float32)
        self.prevCarVel = n.array([0, 0, 0], dtype=n.float32)
        self.prevBallPos = n.array([0, 0, 0], dtype=n.float32)
        self.prevBallVel = n.array([0, 0, 0], dtype=n.float32)
        self.prevDist = 0
        self.touchedBall = False
        self.firstIter = True


    def get_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        # Skip first iteration after reset
        if self.firstIter:
            self.prevBallVel = state.ball.linear_velocity
            self.firstIter = False
            return 0


        ### Order of actions in previous_action
        ### [drive, steer, yaw, pitch, roll, jump, boost, powerslide] ###
        
        # Initialize reward
        reward = 0

        # Car and ball positions
        carPos = player.car_data.position
        carVel = player.car_data.linear_velocity
        ballPos = state.ball.position
        ballVel = state.ball.linear_velocity

        ### START OF REWARDS ###

        # Reward for going toward the ball, bonus for going fast. 1r/s supersonic toward ball
        toBall = ballPos - carPos; toBall /= n.linalg.norm(toBall)
        carVelScalar = n.linalg.norm(carVel)
        angle = n.arccos(n.dot(toBall, carVel / carVelScalar))
        carVelMultiplier = carVelScalar * self.speedMultiplier
        reward += self.faceBallMultiplier * 1/15 * carVelMultiplier / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)


        if player.ball_touched:
            # Reward for touching the ball, higher is better. Double reward if in air
            heightMul = ballPos[2] / (CEILING_Z - 92.75)  # between 0.05 and 1
            reward += 1.4 * self.ballTouchMultiplier * heightMul * (2 - int(player.on_ground))

            # Reward for accelerating the ball, 0 -> supersonic = 1r  ##
            ballDeltaV = ballVel - self.prevBallVel
            reward += self.ballAccelerateMultiplier *  n.linalg.norm(ballDeltaV) / SUPERSONIC_THRESHOLD


            if player.team_num == BLUE_TEAM:
                # Reward for shooting ball on net, supersonic at goal = 1r, 
                ballToGoal = self.orangeGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
                ballVelScalar = n.linalg.norm(ballVel)
                angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
                reward += self.shotOnGoalMultiplier * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Reward for making y-velocity of ball larger
                reward += self.yVelocityReward * ballDeltaV[1] / SUPERSONIC_THRESHOLD
            else:
                # Reward for shooting ball on net, supersonic at goal = 1r, 
                ballToGoal = self.blueGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
                ballVelScalar = n.linalg.norm(ballVel)
                angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
                reward += self.shotOnGoalMultiplier * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Reward for making y-velocity of ball larger
                reward += -self.yVelocityReward * ballDeltaV[1] / SUPERSONIC_THRESHOLD


            # Makes sure no false goal rewards are given
            self.touchedBall = True

        # Update stored values
        self.prevBallVel = ballVel

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        if not self.touchedBall:
            return 0

        reward = 0

        # Reward for scoring in the right goal
        if self.blueScore < state.blue_score:
            self.blueScore += 1
            reward += self.goalReward

        # Punishment for scoring in the wrong goal
        if self.orangeScore < state.orange_score:
            self.orangeScore += 1
            reward -= self.ownGoalReward

        return reward
    

    def _misc_rewards(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:

    
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

        # Initialize reward
        reward = 0

        # Car and ball positions
        carPos = player.car_data.position
        carVel = player.car_data.linear_velocity
        ballPos = state.ball.position
        ballVel = state.ball.linear_velocity

        
        # Reward for being in "good" position. "Perfect" gives 1r/s, worst gives 0.002r/s
        toBall = ballPos - carPos; toBall /= n.linalg.norm(toBall)  # Normalized vetor from car to ball
        if ballPos[1] > 0:  # ball on orange side
            toGoal = self.orangeGoal - carPos; toGoal /= n.linalg.norm(toGoal)
            cosAngle = n.dot(toGoal, toBall)  # cos of angle between vectors
        else:  # Ball on blue side
            toGoal = self.orangeGoal - carPos; toGoal /= n.linalg.norm(toGoal)
            cosAngle = -n.dot(toGoal, toBall)  # cos of angle between vectors
        reward += self.positionMultiplier * 1 / 15 * n.exp(-2 * n.arccos(cosAngle))



        # Reward for facing the ball, (multiplied with carspeed)
        forward = player.car_data.forward()  # Vector in forward direction of car
        toBall = ballPos - carPos; toBall /= n.sqrt(n.dot(toBall, toBall))  # Normalized vetor from car to ball
        speedMultiplier = n.dot(carVel, forward) / 1410  # Car speed, normalized to 1 at max speed w/o boost
        facingMultiplier = n.dot(forward, toBall)
        rew_add = 0.015 * facingMultiplier * speedMultiplier

        if rew_add > 0:
             # Reduce reward for following the ball backwards
            if facingMultiplier < 0 and speedMultiplier < 0:
                rew_add *= 0.7

            reward += rew_add
        else:  # Punish driving away from ball more
            reward += 1.5 * rew_add

        
        # Reward for having the ball go fast toward opposition net
        ballToGoal = self.orangeGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
        ballVelScalar = n.linalg.norm(ballVel)
        if ballVelScalar > 0:
            ballVelNorm = ballVel / ballVelScalar
            reward += 0.1 * n.exp(-30 * n.abs(n.dot(ballVelNorm, ballToGoal) - 1)) * ballVelScalar / 1410

        # Punishment for having the ball go fast toward own net
        ballToGoal = self.blueGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
        # ballVelScalar = n.linalg.norm(ballVel)
        if ballVelScalar > 0:
            # ballVelNorm = ballVel / ballVelScalar
            reward -= 0.1 * n.exp(-30 * n.abs(n.dot(ballVelNorm, ballToGoal) - 1)) * ballVelScalar / 1410
        

        return reward