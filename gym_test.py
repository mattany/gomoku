from gomoku_env import GomokuEnv
import numpy as np
import random
from IPython.display import clear_output
import gym
gym.register('Gomoku-v0', entry_point=GomokuEnv)
env_a = gym.make('Gomoku-v0')
env_b = gym.make('Gomoku-v0')

# for i_episode in range(1000):
#     observation = env_a.reset()
#     for t in range(100):
#         env_a.render()
#         action = env_a.action_space.sample()
#         observation, reward, done, info = env_a.step(action)
#         if done:
#             print(reward)
#             # print("Episode finished after {} timesteps".format(t+1))
#             break
# env_a.close()

alpha = 0.1
gamma = 0.6
epsilon = 0.1
q_table = dict()
wins, loses = 0, 0

num_of_episodes = 100000
board = [[0 for i in range(15)] for j in range(15)]
for episode in range(0, num_of_episodes):
    # Reset the env_a
    state = env_a.reset()

    # Initialize variables
    reward = 0
    terminated = False
    while not terminated:
        if state not in q_table:
            q_table[state] = [0 if env_a.action_space.contains(i) else -np.inf for i in range(225)]
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            action = env_a.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, info = env_a.step(action)
        if terminated:
            if reward == 100:
                wins += 1
            elif reward == -100:
                loses += 1
        if next_state not in q_table:
            q_table[next_state] = [0 if env_a.action_space.contains(i) else -np.inf for i in range(225)]
        # Recalculate
        q_value = q_table[state][action]
        max_value = np.max(q_table[next_state])
        new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

        # Update Q-table
        q_table[state][action] = new_q_value
        state = next_state

    if (episode + 1) % 100 == 0:
        print(wins,loses)
        clear_output(wait=True)
        print("Episode: {}".format(episode + 1))
        env_a.render()

print("**********************************")
print("Training is done!\n")
print("**********************************")