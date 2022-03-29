# import required packages
import numpy as np


# define the function player()
def player(obs, info):
    player = info['player']
    eyes = info['eyes']
    opponents = [0, 1, 2, 3]
    opponents.remove(player)
    for pin in [0, 1, 2, 3]:
        pin_loc = obs[player][pin]
        if pin_loc > 40 or pin_loc == 0:
            continue
        pin_loc += 10 * player - 1
        pin_loc %= 40
        for opponent in opponents:
            for opponent_pin_loc in obs[opponent]:
                if opponent_pin_loc == 0 or opponent_pin_loc > 40:
                    continue
                opponent_pin_loc += 10 * opponent - 1
                opponent_pin_loc %= 40
                if opponent_pin_loc - pin_loc == eyes:
                    return pin
    return obs[player]


def main():
    print('Importing player-file advanced')
    pass


main()
