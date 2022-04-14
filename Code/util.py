import numpy as np
import random
from ludo import random_player

'''GLOBAL CONSTANTS'''
forwards_view_range = 6  # how many tiles it can see forwards
no_forward = 2  # how many opponents it can see forwards
no_pins = 2  # how many of our own pins should be able to see
Qdim = [forwards_view_range + 1] * no_pins * no_forward + [4] * no_pins + [6] + [4]

'''PLAYERS FOR TRAINING'''
our_player = 'our_player'
players = [our_player, random_player, random_player, random_player]

'''PLACE HOLDER GLOBAL VARIABLES'''
player = 0
opponents = [0, 1, 2, 3]


def evaluate_player(state, player_):
    """
    Scores a player based on how far they've progressed
    :param state: state of the board as defined in the project manual
    :param player_: The player to be scored 0-4
    :return: The score of the player
    """
    score = 0
    for pin in [0, 1, 2, 3]:
        pin_index = state[player_][pin]
        if pin_index > 40:
            score += 100  # add score when in home base
        elif pin_index == 0:
            score -= 100  # deduct score when at start
        else:
            score += pin_index  # add score for progress to home base
    return score


def evaluate(state, score_old=0):
    """
    Scores the AI based on the it's own score and the core of the opponents
    :param state: state of the board as defined in the project manual
    :param score_old: The score before the AI made a move
    :return: The score of the AI
    """
    score = 0
    score += 0.003 * evaluate_player(state, player)  # add our own score
    for opponent in opponents:
        score -= 0.001 * evaluate_player(state, opponent)  # subtract our opponents score
    return score - score_old


def winner(state):
    """
    Determines who the winner of the game is
    :param state: state of the board as defined in the project manual
    :return: the winner of the game
    """
    scores = [sum([pos > 40 for pos in s]) for s in state]  # no of pawns in target field for each player
    return np.argsort(scores)[-1]


def start_game(env):
    """
    Ensures the AI has a randomised starting position
    Plays until it is the AI's turn
    :param env: The environment that's being played in
    :return: state, reward, done, info of the last played turn
    """
    global players, player, opponents
    random.shuffle(players)  # randomize the order
    opponents = [0, 1, 2, 3]
    player = players.index(our_player)  # denote who we are
    opponents.remove(player)  # denote who the opponents are
    '''play until it is our move'''
    state, reward, done, info = env.reset()
    while info['player'] != player:
        current_player = players[info['player']]
        state, reward, done, info = env.step(current_player(state, info))
    return state, reward, done, info


def get_index(state, eyes):
    """
    interpret the state and assign indexation indices inside the Q matrix
    :param state: state of the board as defined in the project manual
    :param eyes: eyes of the dice in range 1-6
    :return: Index for Q matrix such that it returns the action probabilities
    """
    state = np.array(state)
    index = np.zeros(len(Qdim) - 1, dtype=int)
    '''get the absolute positions'''
    state_abs = state.copy()
    state_abs += np.array([0, 10, 20, 30]).reshape(-1, 1)
    state_abs -= 1
    state_abs %= 40
    '''split/filter positions'''
    pins_state = state[player]
    pins_state_abs = state_abs[player]
    opponent_state = np.ravel(state[opponents])  # all opponent are treated equally
    opponent_state_abs = np.ravel(state_abs[opponents])
    opponent_state_abs = opponent_state_abs[(opponent_state > 0) * (opponent_state <= 40)]  # remove opponents hb/start
    '''Sort our own pins'''
    pins_start = list(np.argwhere(pins_state < 1).T[0])  # pins at the start
    pins_hb = list(np.argwhere(pins_state > 40).T[0])  # pins in homebase
    pins_bad = pins_start + pins_hb
    pins_good = np.flip(np.argsort(pins_state))  # sort pins in order of closest to homebase
    pins_good = [i for i in pins_good if i not in pins_bad]
    pins = pins_good + pins_bad  # put pins in hb or start to the back
    '''get the distances'''
    for i in range(no_pins):
        if pins[i] in pins_bad:
            continue
        distances = opponent_state_abs - pins_state_abs[pins[i]]
        distances.sort()
        distances = distances[(distances > 0) * (distances < forwards_view_range)]  # apply view distance
        distances = np.append(distances, [0] * no_forward)
        index[i * no_forward:(i + 1) * no_forward] = distances[:no_forward]  # apply view amount
    index[-no_pins - 1:-1] = pins[:no_pins]
    index[-1] = eyes - 1
    return tuple(index)
