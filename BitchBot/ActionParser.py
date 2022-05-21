import numpy as n
import gym.spaces
from rlgym.utils.gamestates import GameState
from rlgym.utils.action_parsers import ActionParser


class BBActionParser(ActionParser):
    # Unused at the current time

    def __init__(self, n_bins=3):
        super().__init__()

        self.possibleActions = []

        # Ground actions
        for drive in [-1, 0, 1]:
            for steer in [-1, 0, 1]:
                for roll in [-1, 0, 1]:
                    for powerslide in [0, 1]:
                        for boost in [0, 1]:
                            if abs(drive) + abs(steer) + abs(roll) + powerslide + boost > 3:
                                continue

                            # KBM simplification
                            pitch = drive
                            yaw = steer

                            # Skip boosting without driving
                            if boost == 1 and drive != 1:
                                continue

                            # Don't need powerslide when not steering or when rolling
                            if powerslide == 1 and (steer == 0 or roll != 0):
                                continue

                            self.possibleActions.append([drive, steer, yaw, pitch, roll, 0, boost, powerslide])

        # Add all types of jumps
        self.possibleActions.append([1, 0, 0, 0, 0, 1, 0, 0])   # Flip forward
        self.possibleActions.append([1, 1, 0, 0, 0, 1, 0, 0])   # Flip diagonally right forward
        self.possibleActions.append([0, 1, 0, 0, 0, 1, 0, 0])   # Flip sideways
        self.possibleActions.append([-1, 1, 0, 0, 0, 1, 0, 0])  # Flip diagonally right backward
        self.possibleActions.append([-1, 0, 0, 0, 0, 1, 0, 0])  # Flip backward
        self.possibleActions.append([-1, -1, 0, 0, 0, 1, 0, 0]) # Flip diagonally left backward
        self.possibleActions.append([0, -1, 0, 0, 0, 1, 0, 0])  # Flip sideways
        self.possibleActions.append([1, -1, 0, 0, 0, 1, 0, 0])  # Flip diagonally left forward
        self.possibleActions.append([0, 0, 0, 0, 0, 1, 0, 0])   # Empty jump

        self.possibleActions = n.array(self.possibleActions)


    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([self.possibleActions.shape[0]])

    def parse_actions(self, actions: n.ndarray, state: GameState) -> n.ndarray:
        if isinstance(actions, n.int64):
            actions = n.array(self.possibleActions[actions], ndmin=2)
        else:
            actions = n.array([[self.possibleActions[action]] for action in actions])

        return actions