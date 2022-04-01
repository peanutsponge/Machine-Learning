import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

N1 = 8
N2 = 10
def create_q_model():
    ## create model instance
    model = Sequential()
    model.add(Dense(N1, input_dim=2, kernel_initializer='normal', activation="relu"))
    model.add(Dense(N2, input_dim=N1, kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_actions,input_dim=N2, kernel_initializer="normal", activation="linear"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def predict(state):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    return model_main.predict(state_tensor)

def QLearn(w, d, ε, ε_min, num_games):
    wins = 0
    ε_adj = (ε - ε_min) / num_games

    for _ in tqdm(range(num_games)):
        state = env.reset()
        Qold = predict(state)
        done = False
        while not done:  # automatically gets satisfied after 200 actions

            if np.random.random() < ε:
                a = env.action_space.sample()  # pick a random
            else:
                a = np.argmax(Qold)  # pick the optimal

            state, reward, done, info = env.step(a)  # execute the action
            Qnew = predict(state)
            won = done and not info
            # Change Q
            Qtarget = reward if won else (1 - w) * Qold[a] + w * (reward + d * np.max(Qnew))
            Qold = Qnew

            model_main.fit(state, Qtarget, verbose=0)
        wins += won
        ε -= ε_adj
    return wins


def Qtest(runs=5000):
    wins = 0
    total_reward_array = np.zeros(runs)
    for i in tqdm(range(runs)):
        state = env.reset()
        done = False
        while not done:
            a = np.argmax(predict(state))
            state, reward, done, info = env.step(a)
            total_reward_array[i] += reward
        wins += not info
    return wins, np.min(total_reward_array), np.mean(total_reward_array), np.max(total_reward_array)


# ---------------GLOBAL VARIABLES-----------------#
env = gym.make('MountainCar-v0')  # selects the type of game
num_actions = env.action_space.n

model_main = create_q_model()
# model_target = create_q_model()
#
# stepsize_main = 4
# stepsize_taget = 100

wins = QLearn(  # Trains a Q-matrix
    w=0.15,  # learning rate
    d=0.95,  # discount rate
    ε=0.80,  # εilon greedy strategy
    ε_min=-3,
    num_games=2000  # number of training games
)

print('Training ended. Number of wins:', wins)
print('Test ended. Number of wins, min reward, avg reward, max reward:', Qtest(runs=500))
env.close()  # close the openai gym environment
