import numpy as n
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER

class BBObservations(ObsBuilder):
    def __init__(self):
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> n.ndarray:
        obs = n.zeros(34, dtype=n.float32)

        # Add total time to observations
        # obs += [self.i / 120.0]

        # Add information about car
        # obs[0:3] = player.car_data.position
        # obs[3:6] = player.car_data.linear_velocity
        # obs[6:9] = player.car_data.forward()
        # obs[9:12] = player.car_data.up()

        # Make the car's orientation as a basis
        carBasis = n.column_stack((player.car_data.forward(), player.car_data.up(), player.car_data.left()))
        CoBMat = n.linalg.inv(carBasis)  # Change of basis matrix

        # Give information about car
        obs[0:3] = player.car_data.forward()  # Give information about orientation of car
        obs[3:6] = player.car_data.up()
        obs[6:9] = CoBMat @ player.car_data.linear_velocity  # Information about current velocity

        # Misc. car data
        obs[10]  = player.boost_amount
        obs[11]  = player.on_ground
        obs[12]  = player.has_jump
        obs[13]  = player.has_flip
        obs[14]  = player.ball_touched

        # Add information about ball (in car's basis)
        obs[14:17] = CoBMat @ (state.ball.position - player.car_data.position)
        obs[17:20] = CoBMat @ state.ball.linear_velocity

        # Add information about goal positions (in car's basis)
        obs[20:23] = CoBMat @ (self.orangeGoal - player.car_data.position)
        obs[23:26] = CoBMat @ (self.blueGoal - player.car_data.position)

        # Add information about previous actions
        obs[26:34] = previous_action

        # Add information about previous action
        # obs[18:26] = previous_action

        # Add player information
        # for player in state.players:
        #     obs += [*player.car_data.position]
        #     obs += [*player.car_data.forward()]
        #     obs += [*player.car_data.linear_velocity]

        # Return data
        return obs