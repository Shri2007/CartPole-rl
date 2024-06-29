import gym
import numpy as np
from agent import DQNAgent

def evaluate_agent(agent, env, n_episodes=10):
    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        for time in range(500):
            env.render()
            action = np.argmax(agent.model(np.array([state]))[0])
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, agent.state_size])
            if done:
                print(f"episode: {e}/10, score: {time}")
                break
    env.close()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load('cartpole_model/cartpole-dqn.h5')
    evaluate_agent(agent, env)