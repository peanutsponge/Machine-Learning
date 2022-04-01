import gym
import numpy as np
from tqdm import tqdm
## load sequential models
from keras.models import Sequential

## load a dense layer all-to-all connection
from keras.layers import Dense, Activation


def StoQ(state):
    # interpret the state and assign location indices inside the Q matrix
    return tuple(np.round((state - Smin) * Qdimn).astype(int))


def create_q_model():
    ## create model instance
    model = Sequential([
        Dense(10, input_dim=Qdim1 + Qdim2),
        Activation('relu'),
        Dense(10),
        Activation('relu'),
        Dense(num_actions)])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


def QLearn(w, d, ε, ε_min, num_games):
    wins = 0
    last_1p_wins = 0

    ε_adj = (ε - ε_min) / num_games

    for p in tqdm(range(num_games)):
        loc = StoQ(env.reset())
        done = False
        while not done:  # automaticaly gets satisfied after 200 actions
            Qold = Q[loc]

            if np.random.random() < ε:
                a = env.action_space.sample()  # pick a random
            else:
                a = np.argmax(Qold)  # pick the optimal

            state, reward, done, info = env.step(a)  # execute the action

            loc_new = StoQ(state)

            won = done and not info
            # Change Q
            Qold[a] = reward if won else (1 - w) * Qold[a] + w * (reward + d * np.max(Q[loc_new]))

            loc = loc_new
        wins += won
        last_1p_wins += won and p >= num_games * 0.9
        ε -= ε_adj

    return wins, last_1p_wins


def Qtest(runs=5000):
    wins = 0
    total_reward_array = np.zeros(runs)
    for i in tqdm(range(runs)):
        state = env.reset()
        done = False
        # counter = 0
        while not done:
            loc = StoQ(state)
            a = np.argmax(Q[loc])
            state, reward, done, info = env.step(a)
            total_reward_array[i] += reward
        wins += not info
    return wins, np.min(total_reward_array), np.mean(total_reward_array), np.max(total_reward_array)


# ---------------GLOBAL VARIABLES-----------------#
env = gym.make('MountainCar-v0')  # selects the type of game
num_actions = env.action_space.n
Smin = env.observation_space.low
Smax = env.observation_space.high
dS = 1 / (Smax - Smin)
Qdim = [15, 10, num_actions]  # dimensions of Q-matrix
Qdim1, Qdim2, _ = Qdim
Qdimn = [Qdim1 - 1, Qdim2 - 1] * dS

model_main = create_q_model()
model_target = create_q_model()

stepsize_main = 4
stepsize_taget = 100

Q = np.random.uniform(low=-1, high=1, size=(Qdim[0], Qdim[1], Qdim[2]))  # random initial Q matrix
wins, last_1p_wins = QLearn(  # Trains a Q-matrix
    w=0.15,  # learning rate
    d=0.95,  # discount rate
    ε=0.80,  # εilon greedy strategy
    ε_min=-3,
    num_games=2000  # number of training games
)

print('Training ended. Number of wins:', wins)
# print(f'Training ended. Number of wins in last {int(0.1 * num_games)}:', last_1p_wins)
print('Test ended. Number of wins, min reward, avg reward, max reward:', Qtest(runs=500))
env.close()  # close the openai gym environment
