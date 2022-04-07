import numpy as np
from tqdm import tqdm
from ludo import make
from ludo import random_player
from ludo import eager_player


def action(Q_old, eps):

    if np.random.random() < eps:
        return np.random.randint(0, 4)  # pick a random
    else:
        return np.argmax(Q_old)  # pick the optimal


def StoQ(state, info):  # interpret the state and assign location indices inside the Q matrix
    player = info['player']
    opponents = [0, 1, 2, 3]
    opponents.remove(player)
    # TODO move above to loop per game for optimisation

    index = np.zeros(5, dtype=int)
    for pin in [0, 1, 2, 3]:
        distances = []
        pin_loc = state[player][pin]
        if pin_loc > 40 or pin_loc == 0:  # player pin is in hb or not on board
            continue
        pin_loc += 10 * player - 1
        pin_loc %= 40
        for opponent in opponents:
            for opponent_pin_loc in state[opponent]:
                if opponent_pin_loc == 0 or opponent_pin_loc > 40:
                    continue
                opponent_pin_loc += 10 * opponent - 1
                opponent_pin_loc %= 40
                distances.append(opponent_pin_loc - pin_loc)
        distances = [x for x in distances if -backwards_view_range <= x <= forwards_view_range]
        if distances:
            distances_p = [x for x in distances if x > 0]
            distances_n = [x for x in distances if x < 0]
            distances_p.sort()
            distances_n.sort(reverse=True)
            d_p = distances_p[0] if distances_p else np.inf
            d_n = distances_n[0] if distances_n else -np.inf
            index[pin] = d_p if d_p <= -d_n else d_n
    index[4] = info['eyes'] - 1
    return Q[tuple(index)]


def QLearn(d, w, eps, eps_min, num_games):
    eps_adj = (eps - eps_min) / num_games
    wins = 0
    for _ in tqdm(range(num_games)):
        state, reward, done, info = env.reset()
        Q_old = StoQ(state, info)
        done = False
        while not done:
            a = action(Q_old, eps)
            state, reward, done, info = env.step(Q_old)  # execute the action
            Q_new = StoQ(state, info)
            if done:  # win condition
                wins += 1
                Q_old[a] = reward
            else:
                Q_old[a] = (1 - w) * Q_old[a] + w * (reward + d * np.max(Q_new))
            Q_old = Q_new

        eps -= eps_adj
    return wins / num_games


def Qtest(num_games=500):
    wins = 0
    for _ in tqdm(range(num_games)):
        state, reward, done, info = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(StoQ(state, info))
        wins += not info
    return wins / num_games


# ---------------GLOBAL VARIABLES-----------------#
backwards_view_range = 6
forwards_view_range = 6

env = make(num_players=4, render=False)
Qdim = [backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        6, 4]  # dimensions of Q-matrix

# ---------------MAIN STUFF-----------------------#
Q = np.random.uniform(low=-1, high=1,
                      size=Qdim)  # random initial Q matrix
print('Training ended. Win rate:',
      QLearn(  # Trains a Q-matrix
          w=0.15,  # learning rate
          d=0.95,  # discount rate
          eps=0.80,  # epsilon greedy strategy
          eps_min=-3,
          num_games=1000))  # number of training games
print('Test ended. Win rate:', Qtest(num_games=100))

env.close()  # close the openai gym environment
