# import required packages
from tqdm import tqdm
from ludo import make
import numpy as np

# import players
from advanced import player as advanced_player
from ludo import random_player
from ludo import eager_player

# constants
PLAYER2COLOR = ['Yellow', 'Red', 'Blue', 'Green']

# create a list of players
players = [advanced_player, random_player, random_player, random_player]

# create list of scores
wins = [0, 0, 0, 0]

# create an instance of the game with 4 players
env = make(num_players=4, render=False)

num_games = 1000

for game in tqdm(range(num_games)):
    # reset the game
    obs, rew, done, info = env.reset()

    # play the game until finsihed
    while not done:
        # get an action from the current player
        current_player = players[info['player']]
        action = current_player(obs, info)

        # pass the action and get the new gamestate
        obs, rew, done, info = env.step(action)

        # render for graphical representation of gamestate
        env.render()

    # compute the winner / ranking
    scores = [sum([pos > 40 for pos in state]) for state in obs]  # no of pawns in target field for each player
    winner = np.argsort(scores)[-1]
    wins[winner] += 1
print(wins)