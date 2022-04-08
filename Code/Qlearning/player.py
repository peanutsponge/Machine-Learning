# import required packages
import numpy as np
import Qlearning.util as util


start = True
Q = 0


# define the function player()
def player(obs, info):
    global start
    if start:
        util.player = info['player']
        util.opponents.remove(util.player)
        start = False
    return Q[util.toIndex(obs, info['eyes'])]


def main():
    global Q
    print('Importing player-file Q')
    Q = np.load("Qlearning/Q.npy")
    pass


main()
