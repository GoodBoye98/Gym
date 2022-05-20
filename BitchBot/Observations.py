import numpy as n
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState

class BBObservations(ObsBuilder):
    def __init__(self):
        self.i = 0


    def reset(self, initial_state: GameState):
        self.i = 0

    def build_obs(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> n.ndarray:

        self.i += 1
        obs = []

        # Add total time to observations
        # obs += [self.i / 120.0]

        # Add ball information
        obs += [*state.ball.position]
        obs += [*state.ball.linear_velocity]
        # Add player information
        for player in state.players:
            obs += [*player.car_data.position]
            obs += [*player.car_data.linear_velocity]

        # Return data
        return n.asarray(obs, dtype=n.float32)