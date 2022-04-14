# import required packages
import numpy as np
import util as util


start = True
Q = None


# define the function player()
def player(obs, info):
    global start
    if start:
        util.player = info['player']
        util.opponents.remove(util.player)
        start = False
    return Q[util.get_index(obs, info['eyes'])]


def main():
    global Q
    print('Importing player-file Group-13')
    Q = np.load("Q.npy")
    pass


main()
