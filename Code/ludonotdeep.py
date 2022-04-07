import numpy as np
import random
from tqdm import tqdm
from ludo import make
from ludo import random_player
from ludo import eager_player


def won(state):
    return sum(state[player]) == 41 + 42 + 43 + 44  # TODO maybe better condition


def start_game():
    global players, player, opponents
    random.shuffle(players)
    opponents = [0, 1, 2, 3]
    player = players.index(our_player)
    opponents.remove(player)

    state, reward, done, info = env.reset()
    while info['player'] != player:
        current_player = players[info['player']]
        state, rew, done, info = env.step(current_player(state, info))
    return state, reward, done, info


def action(index_old, eps):
    if np.random.random() < eps:
        return np.random.randint(-1, 1, 4)  # pick a random
    else:
        return Q[index_old]  # pick the optimal


def toIndex(state, info):  # interpret the state and assign indexation indices inside the Q matrix
    index = np.zeros(5, dtype=int)
    for pin in [0, 1, 2, 3]:
        distances = []
        pin_index = state[player][pin]
        if pin_index > 40 or pin_index == 0:  # player pin is in hb or not on board
            continue
        pin_index += 10 * player - 1
        pin_index %= 40
        for opponent in opponents:
            for opponent_pin_index in state[opponent]:
                if opponent_pin_index == 0 or opponent_pin_index > 40:  # enemy pin is in hb or not on board
                    continue
                opponent_pin_index += 10 * opponent - 1
                opponent_pin_index %= 40
                distances.append(opponent_pin_index - pin_index)
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
    return tuple(index)


def QLearn(d, w, eps, eps_min, num_games):
    eps_adj = (eps - eps_min) / num_games
    wins = 0
    for _ in tqdm(range(num_games)):
        state, reward, done, info = start_game()
        index_old = toIndex(state, info)
        while not done:
            current_player = players[info['player']]
            if current_player != our_player:
                state, rew, done, info = env.step(current_player(state, info))
            else:
                state, reward, done, info = env.step(action(index_old, eps))  # execute the action
                index_new = toIndex(state, info)
                if won(state):
                    wins += 1
                    Q[index_old] = reward
                else:
                    Q[index_old] *= (1 - w)
                    Q[index_old] += w * (reward + d * np.max(Q[index_new]))
                index_old = index_new
        eps -= eps_adj
    return wins / num_games


def Qtest(num_games=500):
    wins = 0
    for _ in tqdm(range(num_games)):
        state, reward, done, info = env.reset()
        while not done:
            state, reward, done, info = env.step(Q[toIndex(state, info)])
        wins += won(state)
    return wins / num_games


# ---------------GLOBAL VARIABLES-----------------#
our_player = 'our_player'
players = [our_player, random_player, random_player, random_player]
player = 0
opponents = [0, 1, 2, 3]
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
