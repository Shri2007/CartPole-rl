import gym
import numpy as np
from agent.py import DQNAgent

def train_agent(agent, env, n_episodes=1000, output_dir='cartpole_model/'):
    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        for time in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{n_episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        if e % 50 == 0:
            agent.save(output_dir + "cartpole-dqn.h5")

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    train_agent(agent, env)