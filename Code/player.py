import numpy as np
import util as util

start = True
Q = None


def player(obs, info):
    """
    Returns t
    :param obs: state of the board as defined in the project manual
    :param info: dictionary containing 'player' 0-4 and 'eyes' 1-6
    :return: list of favorable actions
    """
    global start
    if start:
        util.player = info['player']
        util.opponents.remove(util.player)
        start = False
    return Q[util.get_index(obs, info['eyes'])]


def main():
    """
    initialises the Q matrix
    :return: nothing
    """
    global Q
    print('Importing player-file Group-13')
    Q = np.load("Q.npy")
    pass


main()
