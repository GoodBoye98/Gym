import rlgym
import numpy as n
import torch as th
from stable_baselines3 import PPO
from BitchBot import BBReward, BBObservations, BBActionParser, BBTerminalCondition, BBStateSetter

# Imports for multiple instances
from rlgym.envs import Match
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv


# This is the function we need to provide to our SB3MultipleInstanceEnv to construct a match. Note that this function MUST return a Match object.
def get_match():
    
    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
    return Match(
        reward_function=BBReward(),
        terminal_conditions=BBTerminalCondition(),
        obs_builder=BBObservations(),
        state_setter=DefaultState(),
        action_parser=BBActionParser()
    )


def main():
    # Make the default rlgym environment
    # env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=2, wait_time=20)
    env = rlgym.make(reward_fn=BBReward(), obs_builder=BBObservations(), action_parser=BBActionParser())

    # Initialize PPO from stable_baselines3
    model = PPO("MlpPolicy", env=env, verbose=1, n_steps=256, learning_rate=5e-4, ent_coef=5e-3, batch_size=64)

    # Train the dumb ass agent!
    model.learn(total_timesteps=int(1e6))
    model.save("test_save.zip")

    # Close the environment
    env.close()


if __name__ == '__main__':
    main()