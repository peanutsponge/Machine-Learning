import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def StoQ(state):
    # interpret the state and assign location indices inside the Q matrix
    return tuple(np.round((state - Smin) * Qdimn).astype(int))


def create_q_model():  # copied from https://keras.io/examples/rl/deep_q_network_breakout/
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=Qdim)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


def QLearn(Q, d, w, eps, eps_min, num_games):
    wins = 0
    last_1p_wins = 0

    eps_adj = (eps - eps_min) / num_games

    for p in tqdm(range(num_games)):
        loc = StoQ(env.reset())
        done = False
        while not done:  # automaticaly gets satisfied after 200 actions
            Qold = Q[loc]

            if np.random.random() < eps:
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
        eps -= eps_adj

    return wins, last_1p_wins


def Qtest(Q, runs=5000):
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

w = 0.15  # learning rate
d = 0.95  # discount rate
eps = 0.80  # epsilon greedy strategy
eps_min = -3
num_games = 2000  # number of training games

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()
model_main = create_q_model()
model_target = create_q_model()

stepsize_main = 4
stepsize_taget = 100

Q = np.random.uniform(low=-1, high=1, size=(Qdim[0], Qdim[1], Qdim[2]))  # random initial Q matrix
wins, last_1p_wins = QLearn(Q, d, w, eps, eps_min, num_games)  # Trains a Q-matrix

print('Training ended. Number of wins:', wins)
print(f'Training ended. Number of wins in last {int(0.1 * num_games)}:', last_1p_wins)
print('Test ended. Number of wins, min reward, avg reward, max reward:', Qtest(Q, runs=500))
env.close()  # close the openai gym environment
