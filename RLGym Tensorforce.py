import rlgym
from tqdm import tqdm
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment


def main():
    # Create environment from openAI gym
    env = rlgym.make()

    # Create tensorforce environment
    tensorflow_env = Environment.create(environment=env)

    # Create tensorforce agent
    tensorflow_agent = Agent.create(
        agent='ppo',
        network='auto',
        environment=tensorflow_env,
        batch_size=1,
        learning_rate=5e-4,
        max_episode_timesteps=1000,

        # baseline=dict(type='auto', size=32, depth=1),
        # baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),

        saver={
            'directory': 'data/checkpoints',
            'frequency': 1
        }
    )

    for _ in tqdm(range(10)):
        states = env.reset()
        terminal = False
        while not terminal:
            actions = tensorflow_agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            tensorflow_agent.observe(terminal=terminal, reward=reward)

    tensorflow_env.close()
    env.close()


if __name__ == '__main__':
    main()