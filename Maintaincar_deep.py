import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

num_states = 2
N1 = 24
N2 = 48
def create_q_model():
    ## create model instance
    model = Sequential()
    model.add(Dense(N1, input_dim=num_states, kernel_initializer='normal', activation="relu"))
    model.add(Dense(N2, input_dim=N1, kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_actions, input_dim=N2, kernel_initializer="normal", activation="linear"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


def action(state, ε):
    if np.random.random() < ε:
        a = env.action_space.sample()  # pick a random
    else:
        a = np.argmax(predict(state))  # pick the optimal
    return a


def predict(state):
    return model_main.predict(np.array([state]))[0]


def QLearn(w, d, ε, ε_min, num_games):
    wins = 0
    ε_adj = (ε - ε_min) / num_games
    for _ in tqdm(range(num_games)):
        state = env.reset()
        Qold = predict(state)
        done = False
        while not done:  # automatically gets satisfied after 200 actions
            a = action(state, ε)
            newstate, reward, done, info = env.step(a)  # execute the action
            Qnew = predict(newstate)
            won = done and not info
            # Change Q
            if won:
                Qtarget = reward
            else:
                Qtarget = (1 - w) * Qold + w * (reward + d * np.max(Qnew))
            Qold = Qnew
            model_main.fit(np.array([state]), np.array([Qtarget]), verbose=0)
        wins += won
        ε -= ε_adj
    return wins


def Qtest(runs=5000):
    wins = 0
    for _ in tqdm(range(runs)):
        state = env.reset()
        done = False
        while not done:
            a = np.argmax(predict(state))
            state, reward, done, info = env.step(a)
        wins += not info
    return wins


# ---------------GLOBAL VARIABLES-----------------#
env = gym.make('MountainCar-v0')  # selects the type of game
num_actions = env.action_space.n
tf.compat.v1.disable_eager_execution()
model_main = create_q_model()

wins = QLearn(  # Trains a Q-matrix
    w=0.15,  # learning rate
    d=0.95,  # discount rate
    ε=0.80,  # ε greedy strategy
    ε_min=-3,
    num_games=200  # number of training games
)

print('Training ended. Number of wins:', wins)
print('Test ended. Number of wins, min reward, avg reward, max reward:', Qtest(runs=500))
env.close()  # close the openai gym environment
