import gym
from tqdm import tqdm
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment


def main():
    # Create environment from openAI gym
    env = gym.make('CartPole-v1')

    # Create tensorforce environment
    env = Environment.create(environment=env)

    # Create tensorforce agent
    agent = Agent.create(
        agent='ddpg',
        network='auto',
        environment=env,
        batch_size=50,
        memory=5000,
        learning_rate=5e-4,
        max_episode_timesteps=500,

        saver={
            'directory': 'data/checkpoints',
            'frequency': 10
        }
    )

    # agent = Agent.create(
    #     agent='ppo',
    #     network='auto',
    #     environment=env,
    #     batch_size=5,
    #     learning_rate=5e-4,
    #     max_episode_timesteps=500,
    #     summarizer=dict(directory='summaries', summaries='all'),

    #     baseline=dict(type='auto', size=32, depth=1),
    #     baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),

    #     saver={
    #         'directory': 'data/checkpoints',
    #         'frequency': 10
    #     }
    # )

    # Create tensorforce runner
    runner = Runner(
        agent=agent,
        environment=env
    )


    # Train for 200 episodes
    # runner.run(num_episodes=200)

    # runner.run(num_episodes=100, evaluation=True)

    # act-observer, run for 100 episodes
    for _ in tqdm(range(200)):
        states = env.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

    
    # Evaluate after
    sum_rewards = 0.0
    for _ in tqdm(range(1)):
        states = env.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals,
                independent=True, deterministic=True
            )
            states, terminal, reward = env.execute(actions=actions)
            sum_rewards += reward

    print('Mean episode reward:', sum_rewards / 100)

    # runner.close()
    env.close()

    # OpenAI-Gym environment specification
    # or: environment = Environment.create(
    #         environment='gym', level='CartPole-v1', max_episode_timesteps=500)

    # PPO agent specification
    # agent = dict(
    #     agent='ppo',
    #     # Automatically configured network
    #     network='auto',
    #     # PPO optimization parameters
    #     batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
    #     subsampling_fraction=0.33,
    #     # Reward estimation
    #     likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
    #     reward_processing=None,
    #     # Baseline network and optimizer
    #     baseline=dict(type='auto', size=32, depth=1),
    #     baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
    #     # Regularization
    #     l2_regularization=0.0, entropy_regularization=0.0,
    #     # Preprocessing
    #     state_preprocessing='linear_normalization',
    #     # Exploration
    #     exploration=0.0, variable_noise=0.0,
    #     # Default additional config values
    #     config=None,
    #     # Save agent every 10 updates and keep the 5 most recent checkpoints
    #     saver=dict(directory='model', frequency=10, max_checkpoints=5),
    #     # Log all available Tensorboard summaries
    #     summarizer=dict(directory='summaries', summaries='all'),
    #     # Do not record agent-environment interaction trace
    #     recorder=None
    # )
    # or: Agent.create(agent='ppo', environment=environment, ...)
    # with additional argument "environment" and, if applicable, "parallel_interactions"


if __name__ == '__main__':
    main()