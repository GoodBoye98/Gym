from plistlib import InvalidFileException
import numpy as n
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, SUPERSONIC_THRESHOLD, CEILING_Z, ORANGE_TEAM, BLUE_TEAM
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

class BBReward(RewardFunction):

    def __init__(self):

        # Reward multipliers (values in rewards.cfg)
        self.ballTouchReward         = None      # r per sec touching ball (ground level)
        self.ballAccelerateReward    = None      # r per 0->79.2km/h ball velocity
        self.ballTowardGoal          = None      # r per sec with ball toward goal
        self.goalReward              = None      # r per goal
        self.saveReward              = None      # r per save
        self.demoReward              = None      # r per demo
        self.aquireBoostReward       = None      # r per 0->100 boost
        self.saveBoostReward         = None      # r per sec with 100 boost
        self.rewardShare             = None      # r shared between temmates
        self.opponentNegation        = None      # r negation for avg. opponent rewards
        self.defendingReward         = None      # r for being in defending position
        self.attackingReward         = None      # r for being in attacking position
        self.toDefenceReward         = None      # r for driving toward defense if on wrong side of ball

        # Storing player data
        self.players = {}

        # Storing key values
        self.orangeGoal = ORANGE_GOAL_CENTER
        self.blueGoal = BLUE_GOAL_CENTER

        # Updating rewards
        if not self._updateRewards():
            raise InvalidFileException("invalid values in 'BitchBot/reward.cfg'")

    def _updateRewards(self) -> bool:
        try:
            with open("BitchBot/rewards.cfg", 'r') as file:
                string = ''
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                for line in lines:
                    if len(line) < 1:
                        continue
                    l = line.replace(' ', '').split('=')
                    cmd = l[0]
                    val = float(l[1].split('#')[0])
                    if cmd == 'ballTouchReward':
                        self.ballTouchReward = val
                    elif cmd == 'ballAccelerateReward':
                        self.ballAccelerateReward = val
                    elif cmd == 'ballTowardGoal':
                        self.ballTowardGoal = val
                    elif cmd == 'goalReward':
                        self.goalReward = val
                    elif cmd == 'saveReward':
                        self.saveReward = val
                    elif cmd == 'demoReward':
                        self.demoReward = val
                    elif cmd == 'aquireBoostReward':
                        self.aquireBoostReward = val
                    elif cmd == 'saveBoostReward':
                        self.saveBoostReward = val
                    elif cmd == 'rewardShare':
                        self.rewardShare = val
                    elif cmd == 'opponentNegation':
                        self.opponentNegation = val
                    elif cmd == 'defendingReward':
                        self.defendingReward = val
                    elif cmd == 'attackingReward':
                        self.attackingReward = val
                    elif cmd == 'toDefenceReward':
                        self.toDefenceReward = val
            return True
        except:
            print('Could not update rewards, trying again next reset()')
            return False

    def reset(self, initial_state: GameState):

        # Update reward values
        self._updateRewards()

        self.firstIter = True

        # Reset rewards in playerdata
        for player in self.players.keys():
            self.players[player]['reward'] = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:

        # Initialize player dictionary
        if player not in self.players.keys():
            self.players[player.car_id] = {
                'boost': player.boost_amount,
                'goals': player.match_goals,
                'demos': player.match_demolishes,
                'saves': player.match_saves,
                'team': player.team_num,
                'reward': 0
            }

        # Skip first iteration after each reset-call
        if self.firstIter:
            self.prevBallVel = state.ball.linear_velocity
        
        # Initialize reward
        reward = 0

        # Car and ball positions
        carPos = player.car_data.position
        carVel = player.car_data.linear_velocity
        ballPos = state.ball.position
        ballVel = state.ball.linear_velocity

        ### START OF REWARDS ###

        # Reward for gathering boost
        deltaBoost = player.boost_amount - self.players[player.car_id]['boost']
        if deltaBoost > 0:
            reward += self.aquireBoostReward * deltaBoost / 100

        # Reward for demoing
        reward += self.demoReward * (player.match_demolishes - self.players[player.car_id]['demos'])

        # Reward for saving boost
        reward += 0.0667 * self.saveBoostReward * n.sqrt(player.boost_amount / 100)

        # Reward for saving a shot
        reward += self.saveReward * (player.match_saves - self.players[player.car_id]['saves'])

        # Reward for scoring a goal
        reward += self.goalReward * (player.match_goals - self.players[player.car_id]['goals'])

        # Misc. reward for touching ball
        if player.ball_touched:
            # Reward for touching the ball, higher is better. Double reward if in air
            heightMul = ballPos[2] / (CEILING_Z - 92.75)  # between 0.05 and 1
            reward += self.ballTouchReward * heightMul * 2 * (1.5 - int(player.on_ground))

            # Reward for accelerating the ball, 0 -> supersonic = 1r  ##
            ballDeltaV = ballVel - self.prevBallVel
            ballAceleration = n.linalg.norm(ballDeltaV)
            reward += self.ballAccelerateReward * ballAceleration / SUPERSONIC_THRESHOLD

        # Unit vector pointing toward ball
        toBall = ballPos - carPos; toBall /= n.linalg.norm(toBall)
        ballVelScalar = n.linalg.norm(ballVel) + 1e-6

        if player.team_num == BLUE_TEAM:
            # Reward for having ball go on net
            ballToGoal = self.orangeGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
            angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
            reward += self.ballTowardGoal * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-3 * angle ** 2)

            # 1 when ball is in own goal, 0 in opposition goal
            positionScalar = (5120 - ballPos[1]) / 10240

            # Reward for being between the ball and own net
            toOwnGoal = self.blueGoal - carPos; toOwnGoal /= n.linalg.norm(toOwnGoal)
            angle = n.abs(n.arccos(n.dot(toOwnGoal, toBall)) - n.pi)
            reward += self.defendingReward / 15 * n.exp(-3 * angle ** 2) * positionScalar

            # Reward for being in position to shoot on net
            toGoal = self.orangeGoal - carPos; toGoal /= n.linalg.norm(toGoal)
            angle = n.arccos(n.dot(toGoal, toBall))
            reward += self.attackingReward / 15 * n.exp(-3 * angle ** 2) * (1 - positionScalar)

            # Reward driving toward right side of ball
            if carPos[1] > ballPos[1]:
                reward += self.toDefenceReward * -carVel[1] / SUPERSONIC_THRESHOLD / 15
                reward -= 0.5 * n.abs(carPos[1] - ballPos[1]) / 10240 / 15  # Punish being far away from ball on wrong side
        else:
            # Reward for shooting ball on net, supersonic at goal = 1r, 
            ballToGoal = self.blueGoal - ballPos; ballToGoal /= n.linalg.norm(ballToGoal)
            angle = n.arccos(n.dot(ballToGoal, ballVel / ballVelScalar))
            reward += self.ballTowardGoal * ballVelScalar / SUPERSONIC_THRESHOLD * n.exp(-3 * angle ** 2)

            # 1 when ball is in own goal, 0 in opposition goal
            positionScalar = (5120 + ballPos[1]) / 10240

            # Reward for being between the ball and own net
            toOwnGoal = self.orangeGoal - carPos; toOwnGoal /= n.linalg.norm(toOwnGoal)
            angle = n.abs(n.arccos(n.dot(toOwnGoal, toBall)) - n.pi)
            reward += self.defendingReward / 15 * n.exp(-3 * angle ** 2) * positionScalar

            # Reward for being in position to shoot on net
            toGoal = self.blueGoal - carPos; toGoal /= n.linalg.norm(toGoal)
            angle = n.arccos(n.dot(toGoal, toBall))
            reward += self.attackingReward / 15 * n.exp(-3 * angle ** 2) * (1 - positionScalar)

            # Reward driving toward right side of ball
            if carPos[1] < ballPos[1]:
                reward += self.toDefenceReward * carVel[1] / SUPERSONIC_THRESHOLD / 15
                reward -= 0.5 * n.abs(carPos[1] - ballPos[1]) / 10240 / 15  # Punish being far away from ball on wrong side


        # Update stored values
        self.players[player.car_id]['boost'] = player.boost_amount
        self.players[player.car_id]['goals'] = player.match_goals
        self.players[player.car_id]['demos'] = player.match_demolishes
        self.players[player.car_id]['saves'] = player.match_saves
        self.players[player.car_id]['reward'] += reward
        self.prevBallVel = ballVel

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> float:

        # Reward sharing
        homeTeamReward = 0.0
        otherTeamReward = 0.0
        for p in self.players.keys():
            if self.players[p]['team'] == player.team_num:
                homeTeamReward += self.players[p]['reward']
            else:
                otherTeamReward += self.players[p]['reward']
        # Shares rewards between teammates, and negative reward porportional to what opponents got
        return self.get_reward(player, state, previous_action) + (homeTeamReward - self.players[player.car_id]['reward']) * self.rewardShare - otherTeamReward * 2 * self.opponentNegation / len(self.players)
    

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