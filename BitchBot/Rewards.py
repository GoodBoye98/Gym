import numpy as n
from turtle import speed
from rlgym.utils import math
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, SUPERSONIC_THRESHOLD, CEILING_Z, ORANGE_TEAM, BLUE_TEAM
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self,
        ballTouchReward         = 1.0,      # 1.0
        ballAccelerateReward    = 1.0,      # 1.0
        shotOnGoalReward        = 1.5,      # 1.5
        shotOnOwnGoalReward     = -0.8,     # -0.8
        goalReward              = 2.0,      # 2.0
        ownGoalReward           = -2.0,     # -2.0
        yVelocityReward         = 0.2,      # 0.2
        speedReward             = 0.0,      # 0.0
        towardBallReward        = 0.02,     # 0.02
        saveBoostReward         = 0.15,     # 0.15
        rewardShare             = 0.75,     # 0.75
    ):
        self.orangeScore = 0
        self.blueScore = 0
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)

        ###  REWARD RewardS
        self.ballTouchReward = ballTouchReward                  # r per sec touching ball
        self.ballAccelerateReward = ballAccelerateReward        # r per 0->supersonic
        self.shotOnGoalReward = shotOnGoalReward                # r shot straight at net at supersonic
        self.shotOnOwnGoalReward = shotOnOwnGoalReward          # r shot straight at own net at supersonic
        self.goalReward = goalReward                            # r per goal
        self.ownGoalReward = ownGoalReward                      # r per own goal
        self.speedReward = speedReward                          # r per sec at supersonic
        self.towardBallReward = towardBallReward                # r per sec at ball
        self.yVelocityReward = yVelocityReward                  # r per positive supersonic velocity change in
        self.saveBoostReward = saveBoostReward                  # r per sec with sqrt(boost)
        self.rewardShare = rewardShare                          # r shared between temmates
        ###  REWARD RewardS

        self.orangeReward = 0
        self.blueReward = 0

    def reset(self, initial_state: GameState):
        self.prevCarPos = n.array([0, 0, 0], dtype=n.float32)
        self.prevCarVel = n.array([0, 0, 0], dtype=n.float32)
        self.prevBallPos = n.array([0, 0, 0], dtype=n.float32)
        self.prevBallVel = n.array([0, 0, 0], dtype=n.float32)
        self.prevDist = 0
        self.touchedBall = False
        self.firstIter = True

        self.orangeReward = 0
        self.blueReward = 0


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

        # Reward for moving toward the ball
        toBall = ballPos - carPos; toBall /= n.linalg.norm(toBall)
        carVelScalar = n.linalg.norm(carVel) + 1e-6
        angle = n.arccos(n.dot(toBall, carVel / carVelScalar))
        reward += 0.0667 * self.towardBallReward * n.exp(-2 * angle)

        # # Reward for going fast
        # reward += 0.0667 * self.speedReward * carVelScalar / SUPERSONIC_THRESHOLD

        # Reward for saving boost
        reward += 0.0667 * self.saveBoostReward * n.sqrt(player.boost_amount / 100)

        if player.ball_touched:
            # Reward for touching the ball, higher is better. Double reward if in air
            heightMul = ballPos[2] / (CEILING_Z - 92.75)  # between 0.05 and 1
            reward += 1.4 * self.ballTouchReward * heightMul * (2 - int(player.on_ground))

            # Reward for accelerating the ball, 0 -> supersonic = 1r  ##
            ballDeltaV = ballVel - self.prevBallVel
            reward += self.ballAccelerateReward *  n.linalg.norm(ballDeltaV) / SUPERSONIC_THRESHOLD

            if player.team_num == BLUE_TEAM:
                # Reward for shooting ball on net, supersonic at goal = 1r, 
                ballToGoal = self.orangeGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
                ballVelScalar = n.linalg.norm(ballVel)
                angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
                reward += self.shotOnGoalReward * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Punishment for shooting on own net
                angle = n.abs(angle - n.pi)
                reward += self.shotOnOwnGoalReward * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Reward for making y-velocity of ball larger
                reward += self.yVelocityReward * ballDeltaV[1] / SUPERSONIC_THRESHOLD

                # Reward for being on the right side of the ball
                self.reward += 0.02 / 15 if carPos[1] < ballPos[1] else -0.01 / 15
                self.blueReward += reward  # Setup for reward sharing
            else:
                # Reward for shooting ball on net, supersonic at goal = 1r, 
                ballToGoal = self.blueGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
                ballVelScalar = n.linalg.norm(ballVel)
                angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
                reward += self.shotOnGoalReward * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Punishment for shooting on own net
                angle = n.abs(angle - n.pi)
                reward += self.shotOnOwnGoalReward * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-2 * angle)

                # Reward for making y-velocity of ball smaller
                reward += -self.yVelocityReward * ballDeltaV[1] / SUPERSONIC_THRESHOLD

                # Reward for being on the right side of the ball
                self.reward += 0.02 / 15 if carPos[1] > ballPos[1] else -0.01 / 15
                self.orangeReward += reward  # Setup for reward sharing


            # Makes sure no false goal rewards are given
            self.touchedBall = True

        # Update stored values
        self.prevBallVel = ballVel

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:
        if not self.touchedBall:
            return 0

        # Initialize with shared reward values
        reward = self.blueReward if player.team_num == BLUE_TEAM else self.orangeReward
        reward *= self.rewardShare

        # Reward for scoring in the right goal
        if self.blueScore < state.blue_score:
            self.blueScore += 1
            if player.team_num == BLUE_TEAM:
                reward += self.goalReward
            else:
                reward -= self.goalReward

        # Punishment for scoring in the wrong goal
        if self.orangeScore < state.orange_score:
            self.orangeScore += 1
            if player.team_num == BLUE_TEAM:
                reward -= self.goalReward
            else:
                reward += self.goalReward

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
        reward += self.positionReward * 1 / 15 * n.exp(-2 * n.arccos(cosAngle))



        # Reward for facing the ball, (multiplied with carspeed)
        forward = player.car_data.forward()  # Vector in forward direction of car
        toBall = ballPos - carPos; toBall /= n.sqrt(n.dot(toBall, toBall))  # Normalized vetor from car to ball
        speedReward = n.dot(carVel, forward) / 1410  # Car speed, normalized to 1 at max speed w/o boost
        facingReward = n.dot(forward, toBall)
        rew_add = 0.015 * facingReward * speedReward

        if rew_add > 0:
             # Reduce reward for following the ball backwards
            if facingReward < 0 and speedReward < 0:
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