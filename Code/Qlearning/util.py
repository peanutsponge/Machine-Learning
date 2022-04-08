import numpy as np
import random
from Code.ludo import random_player, eager_player

our_player = 'our_player'
players = [our_player, random_player, random_player, random_player]
player = 0
opponents = [0, 1, 2, 3]
backwards_view_range = 0
forwards_view_range = 3

Qdim = [backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        backwards_view_range + forwards_view_range + 1,
        6, 4]  # dimensions of Q-matrix


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
    index[4] = eyes - 1
    return tuple(index)
