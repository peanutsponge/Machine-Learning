from tqdm import tqdm
from ludo import make
from util import *


def action(index_old, eps):
    if np.random.random() < eps:
        return np.random.randint(-1, 1, 4)  # pick a random
    else:
        return Q[index_old]  # pick the optimal


def learn(d, w, eps, eps_min, num_games):
    eps_adj = (eps - eps_min) / num_games
    wins = 0
    for _ in tqdm(range(num_games)):
        state, _, done, info = start_game(env)
        while not done:
            current_player = players[info['player']]
            if current_player != our_player:
                state, _, done, info = env.step(current_player(state, info))
            else:
                reward = evaluate(state)
                eyes = info['eyes']
                index_old = get_index(state, eyes)
                state, _, done, info = env.step(action(index_old, eps))  # execute the action
                index_new = get_index(state, eyes)
                reward = evaluate(state, reward)

                if done and winner(state) == player:
                    Q[index_old] = reward
                else:
                    Q[index_old] *= (1 - w)
                    Q[index_old] += w * (reward + d * np.max(Q[index_new]))
        if winner(state) == player:
            wins += 1
        eps -= eps_adj
    return wins / num_games


env = make(num_players=4, render=False)
Q = np.random.uniform(low=-1, high=1, size=Qdim)  # random initial Q matrix
print('Training ended. Win rate:',
      learn(  # Trains a Q-matrix
          w=0.15,  # learning rate
          d=0.95,  # discount rate
          eps=0.80,  # epsilon greedy strategy
          eps_min=-3,
          num_games=10000))  # number of training games
np.save("Q", Q)


