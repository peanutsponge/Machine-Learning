from ludo import make
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

num_states = 4
N1 = 8
N2 = 10


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




def to_state(obs, info):
    player = info['player']
    opponents = [0, 1, 2, 3]
    opponents.remove(player)
    # TODO move above to loop per game for optimisation

    state = [info['eyes']]
    for pin in [0, 1, 2, 3]:
        distances = [np.inf] * no_distances_per_pin  # TODO Check whether np.inf is good
        pin_loc = obs[player][pin]
        state.append(pin_loc)  # TODO Check whether this is good
        if pin_loc > 40 or pin_loc == 0:  # player pin is in hb or not on board
            continue
        pin_loc += 10 * player - 1
        pin_loc %= 40
        for opponent in opponents:
            for opponent_pin_loc in obs[opponent]:
                if opponent_pin_loc == 0 or opponent_pin_loc > 40:
                    continue
                opponent_pin_loc += 10 * opponent - 1
                opponent_pin_loc %= 40
                distances.append(opponent_pin_loc - pin_loc)
        distances.sort()
        state.append(distances[:no_distances_per_pin])
    return state


def predict(state):
    return model_main.predict(np.array([state]))[0]


def QLearn(w, d, ε, ε_min, num_games):
    wins = 0
    ε_adj = (ε - ε_min) / num_games

    for _ in tqdm(range(num_games)):
        state = env.reset()

        done = False
        while not done:  # automatically gets satisfied after 200 actions
            a = action(state, ε)
            newstate, reward, done, info = env.step(a)  # execute the action
            won = done and not info
            # Change Q
            if won:
                Qtarget = reward
            else:
                Qold = predict(newstate)
                Qnew = predict(state)
                Qtarget = (1 - w) * Qold + w * (reward + d * np.max(Qnew))
                state = newstate
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
# create an instance of the game with 4 players
env = make(num_players=4)
num_actions = env.action_space.n
tf.compat.v1.disable_eager_execution()
model_main = create_q_model()

no_distances_per_pin = 4

wins = QLearn(  # Trains a Q-matrix
    w=0.15,  # learning rate
    d=0.95,  # discount rate
    ε=0.80,  # ε greedy strategy
    ε_min=-3,
    num_games=1000  # number of training games
)

print('Training ended. Number of wins:', wins)
print('Test ended. Number of wins, min reward, avg reward, max reward:', Qtest(runs=500))
env.close()  # close the openai gym environment
