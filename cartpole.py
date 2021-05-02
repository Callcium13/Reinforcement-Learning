import gym
from DQN_with_notes import Agent
from toolbox import plotlearning
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[4], lr=0.001)
    epRewards, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        epReward = 0
        done = False
        # noinspection PyRedeclaration
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            epReward += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        epRewards.append(epReward)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(epRewards[-100:])

        print('episode ', i, 'score %.2f' % epReward,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i + 1 for i in range(n_games)]
    filename = 'cartpole.png'
    plotlearning(x, epRewards, eps_history, filename)
