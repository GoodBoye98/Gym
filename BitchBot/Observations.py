import numpy as n
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER, ORANGE_TEAM, BLUE_TEAM, BOOST_LOCATIONS

class BBObservations(ObsBuilder):
    def __init__(self):
        self.orangeGoal = n.array(ORANGE_GOAL_CENTER, dtype=n.float32)
        self.blueGoal   = n.array(BLUE_GOAL_CENTER, dtype=n.float32)
        self.boostLocationsBlue   = n.array(BOOST_LOCATIONS)
        self.boostLocationsOrange = n.array(BOOST_LOCATIONS)
        self.normalizing = 1000
        self.flip = n.array([-1, -1, 1])

        for i in range(self.boostLocationsOrange.shape[0]):
            self.boostLocationsOrange[i] *= self.flip

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: n.ndarray) -> n.ndarray:
        # Reserve values in array
        obs = n.zeros(34 + 15 * (len(state.players) - 1) + 3 * self.boostLocationsBlue.shape[0], dtype=n.float32)

        # Make the car's orientation as a basis
        carBasis = n.column_stack((player.car_data.forward(), player.car_data.up(), player.car_data.left()))
        CoBMat = n.linalg.inv(carBasis) / self.normalizing  # Change of basis matrix, normalized-ish

        # Add information about ball (in car's basis)
        obs[0:3] = CoBMat @ (state.ball.position - player.car_data.position)
        obs[3:6] = CoBMat @ state.ball.linear_velocity

        # Add information about previous actions
        obs[12:20] = previous_action
        obs[26:29] = CoBMat @ player.car_data.linear_velocity  # Current velocity

        # Misc. car data
        obs[29]  = player.boost_amount / 100
        obs[30]  = player.on_ground
        obs[31]  = player.has_jump
        obs[32]  = player.has_flip
        obs[33]  = player.ball_touched

        # Flipt orientation vectors for orange team
        flip = self.flip if player.team_num == ORANGE_TEAM else 1

        # Give information about teammates car
        i = 0
        for other_player in state.players:
            if player.team_num != other_player.team_num or player == other_player:
                continue
            # Give information about opponents or teammates
            obs[34+i*15:37+i*15] = CoBMat @ (other_player.car_data.position - player.car_data.position)
            obs[37+i*15:40+i*15] = other_player.car_data.forward() * flip
            obs[40+i*15:43+i*15] = other_player.car_data.up() * flip
            obs[43+i*15:46+i*15] = CoBMat @ other_player.car_data.linear_velocity

            # Misc. car data
            obs[46+i*15]  = other_player.boost_amount / 100
            obs[47+i*15]  = other_player.on_ground
            obs[48+i*15]  = other_player.has_flip
            i += 1
        
        # Give information about opponents cars
        for other_player in state.players:
            if player.team_num == other_player.team_num:
                continue
            # Give information about opponents or teammates
            obs[34+i*15:37+i*15] = CoBMat @ (other_player.car_data.position - player.car_data.position)
            obs[37+i*15:40+i*15] = other_player.car_data.forward() * flip
            obs[40+i*15:43+i*15] = other_player.car_data.up() * flip
            obs[43+i*15:46+i*15] = CoBMat @ other_player.car_data.linear_velocity

            # Misc. car data
            obs[46+i*15]  = other_player.boost_amount / 100
            obs[47+i*15]  = other_player.on_ground
            obs[48+i*15]  = other_player.has_flip
            i += 1

        # Add information about goal positions (in car's basis)
        if player.team_num == BLUE_TEAM:  # Set opponent goal at 6-9 and own goal at 9-12
            obs[6:9] = CoBMat @ (self.orangeGoal - player.car_data.position)
            obs[9:12] = CoBMat @ (self.blueGoal - player.car_data.position)

            # Give information about self
            obs[20:23] = player.car_data.forward()  # Orientation of car
            obs[23:26] = player.car_data.up()  # Orientation of car

            # Give location of boost-pads
            for j, boost in enumerate(self.boostLocationsBlue):
                obs[34+i*15+j*3:37+i*15+j*3] = CoBMat @ (boost - player.car_data.position)

        if player.team_num == ORANGE_TEAM:  # Set opponent goal at 6-9 and own goal at 9-12
            obs[6:9] = CoBMat @ (self.blueGoal - player.car_data.position)
            obs[9:12] = CoBMat @ (self.orangeGoal - player.car_data.position)

            # Give information about self
            obs[20:23] = player.car_data.forward() * flip # Orientation of car, x & y-axis flipped
            obs[23:26] = player.car_data.up() * flip # Orientation of car, x & y-axis flipped
            
            # Give location of boost-pads
            for j, boost in enumerate(self.boostLocationsOrange):
                obs[34+i*15+j*3:37+i*15+j*3] = CoBMat @ (boost - player.car_data.position)


        # Return data
        return obs