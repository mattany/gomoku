total_epochs = 0
total_penalties = 0
num_of_episodes = 100

for _ in range(num_of_episodes):
    state = enviroment.reset()
    epochs = 0
    penalties = 0
    reward = 0

    terminated = False

    while not terminated:
        action = np.argmax(q_table[state])
        state, reward, terminated, info = enviroment.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("**********************************")
print("Results")
print("**********************************")
print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
print("Penalties per episode: {}".format(total_penalties / num_of_episodes))