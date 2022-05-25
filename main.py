import rlgym
import numpy as n
import torch as th
from stable_baselines3 import PPO, ddpg
from stable_baselines3.common.callbacks import CheckpointCallback
from BitchBot import BBReward, BBObservations, BBActionParser, BBStateSetter, BBTerminalCondition

# Imports for multiple instances
from rlgym.envs import Match
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

N_STEPS = 512
BATCH_SIZE = 256


# This is the function we need to provide to our SB3MultipleInstanceEnv to construct a match. Note that this function MUST return a Match object.
def get_match():
    
    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
    return Match(
        reward_function=BBReward(),
        obs_builder=BBObservations(),
        state_setter=BBStateSetter(),
        action_parser=BBActionParser(),
        terminal_conditions=BBTerminalCondition(N_STEPS)
    )


def main():
    # Make the default rlgym environment
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=6, wait_time=20, force_paging=True)
    # env = rlgym.make(
    #     reward_fn=BBReward(),
    #     obs_builder=BBObservations(),
    #     state_setter=BBStateSetter(),
    #     action_parser=BBActionParser(),
    #     terminal_conditions=BBTerminalCondition(N_STEPS)
    # )

    # Initialize PPO from stable_baselines3
    model = PPO("MlpPolicy", env=env, verbose=1, n_steps=N_STEPS, batch_size=BATCH_SIZE, learning_rate=5e-5, ent_coef=0.01)
    model.set_parameters("init.zip")  # Load parameters from earlier run

    # Save checkpoints
    callback = CheckpointCallback(round(1e6 / env.num_envs), save_path="Iterations", name_prefix="bb_iteration")
    # callback = CheckpointCallback(1e6, save_path="Iterations", name_prefix="bb_iteration")

    # Train the agent!
    model.learn(total_timesteps=int(1e8), callback=callback)


if __name__ == '__main__':
    main()
