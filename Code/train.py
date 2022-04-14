from tqdm import tqdm
from ludo import make
from util import *


def action(index, eps):
    """
    Takes either a random action or th optimal action accoriding to the Qmatrix
    :param index: The index of the q matrix to choose the action from
    :param eps: The randomness factor - higher is more random
    :return: An action 0-4
    """
    if np.random.random() < eps:
        return np.random.randint(-1, 1, 4)  # pick a random
    else:
        return Q[index]  # pick the optimal


def learn(d, w, eps, eps_min, num_games):
    """
    Trains the AI
    :param d: The discount factor - higher
    :param w: The learning rate  - higher is bigger learning steps
    :param eps: The start randomness factor - higher is more random
    :param eps_min: The randomness factor at the end of the training - negative means the randomness will stop partway
    :param num_games: The amount of games to train for
    :return: The win rate during the training
    """
    eps_adj = (eps - eps_min) / num_games  # calculate randomness adjustment factor
    wins = 0
    for _ in tqdm(range(num_games)):
        state, _, done, info = start_game(env)
        while not done:
            current_player = players[info['player']]
            if current_player != our_player:  # opponent turn
                state, _, done, info = env.step(current_player(state, info))
            else:
                '''Our turn'''
                eyes = info['eyes']
                reward = evaluate(state)  # score of board before move
                index_old = get_index(state, eyes)
                state, _, done, info = env.step(action(index_old, eps))  # execute the action
                index_new = get_index(state, eyes)
                reward = evaluate(state, reward)  # score of board after move - score of board before move
                '''Apply Bell's equation'''
                if done and winner(state) == player:
                    Q[index_old] = reward
                else:
                    Q[index_old] *= (1 - w)
                    Q[index_old] += w * (reward + d * np.max(Q[index_new]))
        if winner(state) == player:  # update scores
            wins += 1
        eps -= eps_adj  # adjust randomness factor
    return wins / num_games


'''CONTROL MENU'''
env = make(num_players=4, render=False)
Q = np.random.uniform(low=-1, high=1, size=Qdim)  # random initial Q matrix
print('Training ended. Win rate:',
      learn(  # Trains a Q-matrix
          w=0.15,  # learning rate
          d=0.95,  # discount rate
          eps=0.80,  # epsilon greedy strategy
          eps_min=-3,  # epsilon greedy strategy minimum
          num_games=10000))  # number of training games
np.save("Q", Q)
