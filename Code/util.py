import numpy as np
import random
from ludo import random_player, eager_player

our_player = 'our_player'
players = [our_player, random_player, random_player, random_player]
player = 0
opponents = [0, 1, 2, 3]

backwards_view_range = 2  # how many tiles it can see backwards
forwards_view_range = 6  # how many tiles it can see forwards
no_forward = 3  # how many opponents it can see forwards
no_backward = 0  # how many opponents it can see backwards
no_pins = 1  # how many of our own pins should be able to see

Qdim = [forwards_view_range + 1] * no_pins * no_forward + \
       [backwards_view_range + 1] * no_pins * no_backward + \
       [4] * no_pins + \
       [6] + \
       [4]  # the number of actions


def evaluate_player(state, player_):
    score = 0
    for pin in [0, 1, 2, 3]:
        pin_index = state[player_][pin]
        if pin_index > 40:
            score += 100
        elif pin_index == 0:
            score -= 100
        else:
            score += pin_index
    return score


def evaluate(state, reward_old=0):
    score = 0
    score += 0.003 * evaluate_player(state, player)
    for opponent in opponents:
        score -= 0.001 * evaluate_player(state, opponent)
    return score - reward_old


def winner(state):
    scores = [sum([pos > 40 for pos in s]) for s in state]  # no of pawns in target field for each player
    return np.argsort(scores)[-1]


def start_game(env):
    global players, player, opponents
    random.shuffle(players)
    opponents = [0, 1, 2, 3]
    player = players.index(our_player)
    opponents.remove(player)

    state, reward, done, info = env.reset()
    while info['player'] != player:
        current_player = players[info['player']]
        state, reward, done, info = env.step(current_player(state, info))
    return state, reward, done, info


def toIndex(state, eyes):  # interpret the state and assign indexation indices inside the Q matrix
    index = np.zeros(len(Qdim) - 1, dtype=int)
    pins = np.argsort(state[player])  # TODO pins that are in the home are currently used -_-
    for i in range(no_pins):
        distances_p, distances_n = [], []
        pin_index = state[player][pins[i]]
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
                distance = opponent_pin_index - pin_index
                if 0 < distance < forwards_view_range:
                    distances_p.append(distance)
                elif 0 < -distance < -backwards_view_range:
                    distances_p.append(-distance)
            distances_p.sort()
            distances_n.sort()
            distances_p += [0] * no_forward
            distances_n += [0] * no_backward
            index[i * no_forward:(i + 1) * no_forward] = distances_p[:no_forward]
            index[(i + 1) * no_forward:(i + 1) * no_forward + no_backward] = distances_n[:no_backward]
    index[-no_pins - 1:-1] = pins[:no_pins]
    index[-1] = eyes - 1
    return tuple(index)
