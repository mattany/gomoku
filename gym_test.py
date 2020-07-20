import gomoku_env
from gomoku_env import GomokuEnv, board_domination_heuristic
import numpy as np
import random
from IPython.display import clear_output
import gym
gym.register('Gomoku-v0', entry_point=GomokuEnv)
env = gym.make('Gomoku-v0')

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

verbose = False

a_q_table = dict()
b_q_table = dict()
wins, losses = 0, 0

num_of_episodes = 100000
for episode in range(0, num_of_episodes):

    # Reset the env_a
    a_state = env.reset()
    b_state = None
    # Initialize variables
    a_reward = 0
    b_reward = 0
    terminated = False
    while not terminated:
        if a_state not in a_q_table:
            a_q_table[a_state] = [0 if env.action_space.contains(i) else -np.inf for i in range(225)]
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            a_action = env.action_space.sample()
        else:
            a_action = np.argmax(a_q_table[a_state])

        # Take action
        next_b_state, b_reward, terminated, info = env.step(a_action, gomoku_env.PLAYER1)
        if verbose:
            print(env.observation_space)
        if terminated:
            if b_reward == -100:
                wins += 1
        if next_b_state not in b_q_table:
            b_q_table[next_b_state] = [0 if env.action_space.contains(i) else -np.inf for i in range(225)]
        # Recalculate
        if b_state is not None:
            b_q_value = b_q_table[b_state][b_action]
            max_b_value = np.max(b_q_table[next_b_state])
            new_b_q_value = (1 - alpha) * b_q_value + alpha * (b_reward + gamma * max_b_value)

            # Update Q-table
            b_q_table[b_state][b_action] = new_b_q_value
        b_state = next_b_state
        if terminated:
            break

        if b_state not in b_q_table:
            b_q_table[b_state] = [0 if env.action_space.contains(i) else -np.inf for i in range(225)]
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            b_action = env.action_space.sample()
        else:
            b_action = np.argmax(b_q_table[b_state])

        # Take action
        next_a_state, a_reward, terminated, info = env.step(b_action, gomoku_env.PLAYER2)
        if verbose:
            print(env.observation_space)
        if terminated:
            if a_reward == -100:
                losses += 1
        if next_a_state not in a_q_table:
            a_q_table[next_a_state] = [0 if env.action_space.contains(i) else -np.inf for i in range(225)]
        # Recalculate
        a_q_value = a_q_table[a_state][a_action]
        max_a_value = np.max(a_q_table[next_a_state])
        new_q_value = (1 - alpha) * a_q_value + alpha * (b_reward + gamma * max_a_value)

        # Update Q-table
        a_q_table[a_state][a_action] = new_q_value
        a_state = next_a_state










    if (episode + 1) % 100 == 0:
        print(wins,losses)
        clear_output(wait=True)
        print("Episode: {}".format(episode + 1))
        env.render()

    if (episode + 1) % 1500 == 0:
        verbose = True
    if (episode + 1) % 101 == 0:
        verbose = False


print("**********************************")
print("Training is done!\n")
print("**********************************")