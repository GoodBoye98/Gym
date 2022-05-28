import numpy as n
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, ORANGE_TEAM, BLUE_TEAM

class BBObservations(ObsBuilder):
    def __init__(self):
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> n.ndarray:
        # Reserve values in array
        obs = n.zeros(20 + 14 * len(state.players), dtype=n.float32)

        # Make the car's orientation as a basis
        carBasis = n.column_stack((player.car_data.forward(), player.car_data.up(), player.car_data.left()))
        CoBMat = n.linalg.inv(carBasis)  # Change of basis matrix

        # Add information about ball (in car's basis)
        obs[0:3] = CoBMat @ (state.ball.position - player.car_data.position)
        obs[3:6] = CoBMat @ state.ball.linear_velocity

        # Add information about goal positions (in car's basis)
        if player.team_num == BLUE_TEAM:  # Set opponent goal at 6-9 and own goal at 9-12
            obs[6:9] = CoBMat @ (self.orangeGoal - player.car_data.position)
            obs[9:12] = CoBMat @ (self.blueGoal - player.car_data.position)
        if player.team_num == ORANGE_TEAM:  # Set opponent goal at 6-9 and own goal at 9-12
            obs[6:9] = CoBMat @ (self.blueGoal - player.car_data.position)
            obs[9:12] = CoBMat @ (self.orangeGoal - player.car_data.position)

        # Add information about previous actions
        obs[12:20] = previous_action

        # Give information about self
        obs[20:23] = player.car_data.forward()  # Orientation of car
        obs[23:26] = player.car_data.up()  # Orientation of car
        obs[26:29] = CoBMat @ player.car_data.linear_velocity  # Current velocity

        # Misc. car data
        obs[29]  = player.boost_amount
        obs[30]  = player.on_ground
        obs[31]  = player.has_jump
        obs[32]  = player.has_flip
        obs[33]  = player.ball_touched

        # Give information about all other car(s)
        i = 1
        for other_player in state.players:
            if player == other_player:
                continue
            # Give information about opponents or teammates
            obs[20+i*14:23+i*14] = CoBMat @ other_player.car_data.forward()
            obs[23+i*14:26+i*14] = CoBMat @ other_player.car_data.up()
            obs[26+i*14:29+i*14] = CoBMat @ other_player.car_data.linear_velocity

            # Misc. car data
            obs[29+i*14]  = other_player.boost_amount
            obs[30+i*14]  = other_player.on_ground
            obs[31+i*14]  = other_player.has_jump
            obs[32+i*14]  = other_player.has_flip
            obs[33+i*14]  = other_player.ball_touched
            i += 1


        # Return data
        return obs