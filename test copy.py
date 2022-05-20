import gym
from tqdm import tqdm
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment


def main():
    # Create environment from openAI gym
    env = gym.make('CartPole-v1')

    # Create tensorforce environment
    environment = Environment.create(environment=env)
    
    # Load pre-trained agent
    agent = Agent.load(directory='data/checkpoints', format='checkpoint', environment=environment)

    # Show how it does
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        env.render(mode='human')
        actions, internals = agent.act(
            states=states, internals=internals,
            independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)

    environment.close()
    env.close()


if __name__ == '__main__':
    main()