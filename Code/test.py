from tqdm import tqdm
from ludo import make
from player import player as q_player
from ludo import random_player
from util import winner

players = [random_player, random_player, q_player, random_player]  # create a list of players
num_games = 10000
wins = [0, 0, 0, 0]  # create list of scores
env = make(num_players=4, render=False)  # create an instance of the game with 4 players
for game in tqdm(range(num_games)):
    obs, _, done, info = env.reset()  # reset the game
    while not done:  # play the game until finished
        current_player = players[info['player']]  # get an action from the current player
        action = current_player(obs, info)
        obs, _, done, info = env.step(action)  # pass the action and get the new game-state
    wins[winner(obs)] += 1  # compute the winner / ranking
print(wins)
