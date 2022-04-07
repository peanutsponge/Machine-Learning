import gym
import numpy as np
from tqdm import tqdm


def StoQ(state):  # interpret the state and assign location indices inside the Q matrix
    return Q[tuple(np.round((state - Smin) * Qdimn).astype(int))]


def action(Q_old, eps):
    if np.random.random() < eps:
        return env.action_space.sample()  # pick a random
    else:
        return np.argmax(Q_old)  # pick the optimal


def QLearn(d, w, eps, eps_min, num_games):
    eps_adj = (eps - eps_min) / num_games
    wins = 0
    for _ in tqdm(range(num_games)):
        Q_old = StoQ(env.reset())
        done = False
        while not done:
            a = action(Q_old, eps)
            state, reward, done, info = env.step(a)  # execute the action
            Q_new = StoQ(state)
            if done and not info:  # win condition
                wins += 1
                Q_old[a] = reward
            else:
                Q_old[a] = (1 - w) * Q_old[a] + w * (reward + d * np.max(Q_new))
            Q_old = Q_new
        eps -= eps_adj
    return wins / num_games


def Qtest(num_games=500):
    wins = 0
    for _ in tqdm(range(num_games)):
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(np.argmax(StoQ(state)))
        wins += not info
    return wins / num_games


# ---------------GLOBAL VARIABLES-----------------#
env = gym.make('MountainCar-v0')  # selects the type of game
Qdim = [15, 10, env.action_space.n]  # dimensions of Q-matrix
Smin = env.observation_space.low
Smax = env.observation_space.high
dS = 1 / (Smax - Smin)
Qdim1, Qdim2, _ = Qdim
Qdimn = [Qdim1 - 1, Qdim2 - 1] * dS

# ---------------MAIN STUFF-----------------------#
Q = np.random.uniform(low=-1, high=1, size=(Qdim[0], Qdim[1], Qdim[2]))  # random initial Q matrix
print('Training ended. Win rate:',
      QLearn(  # Trains a Q-matrix
          w=0.15,  # learning rate
          d=0.95,  # discount rate
          eps=0.80,  # epsilon greedy strategy
          eps_min=-3,
          num_games=1000))  # number of training games
print('Test ended. Win rate:', Qtest(num_games=100))

env.close()  # close the openai gym environment
