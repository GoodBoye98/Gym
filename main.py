import numpy as n
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from BitchBot import BBReward, BBObservations, BBActionParser, BBStateSetter, BBTerminalCondition

# Imports for multiple instances
from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

N_STEPS = 1024
BATCH_SIZE = 512


# Construct matches
def get_match():
    
    # Configuration
    return Match(
        reward_function=BBReward(),
        obs_builder=BBObservations(),
        state_setter=BBStateSetter(),
        action_parser=BBActionParser(),
        terminal_conditions=BBTerminalCondition(N_STEPS),
        game_speed=100,
        spawn_opponents=True
    )


def main():
    # Make the vectorized rlgym environment
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=10, wait_time=20)

    # Initialize PPO from SB3
    model = PPO("MlpPolicy", env=env, verbose=1, n_steps=N_STEPS, batch_size=BATCH_SIZE, learning_rate=1e-4, ent_coef=0.01)
    model.set_parameters("init")  # Load parameters from init

    # Save checkpoints
    callback = CheckpointCallback(round(1e6 / env.num_envs), save_path="Iterations", name_prefix="bb_iteration")

    # Train! (10 000 000 = 1e7 ~ 1 hr training)
    model.learn(total_timesteps=int(24e7), callback=callback)


if __name__ == '__main__':
    main()
