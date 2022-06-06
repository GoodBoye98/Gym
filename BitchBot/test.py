import time

class Test():

    def __init__(self):

        # Reward multipliers
        self.ballTouchReward         = 0.0      # r per sec touching ball (ground level)
        self.ballAccelerateReward    = 0.0      # r per 0->79.2km/h ball velocity
        self.ballTowardGoal          = 0.0      # r per sec with ball toward goal
        self.goalReward              = 0.0      # r per goal
        self.saveReward              = 0.0      # r per save
        self.demoReward              = 0.0      # r per demo
        self.aquireBoostReward       = 0.0      # r per 0->100 boost
        self.saveBoostReward         = 0.0      # r per sec with 100 boost
        self.rewardShare             = 0.0      # r shared between temmates
        self.opponentNegation        = 0.0      # r negation for avg. opponent rewards
        self.defendingReward         = 0.0      # r for being in defending position
        self.attackingReward         = 0.0      # r for being in attacking position
        self.toDefenceReward         = 0.0      # r for driving toward defense if on wrong side of ball

        # Updating rewards
        self._updateRewards()

    def _updateRewards(self):
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
            print('ballTouchReward =', self.ballTouchReward)
        except:
            print('Could not update rewards, trying again next reset()')


if __name__ == '__main__':
    cl = Test()

    while True:
        cl._updateRewards()
        time.sleep(0.1)