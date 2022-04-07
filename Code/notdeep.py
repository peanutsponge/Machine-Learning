import gym
import numpy as np
from tqdm import tqdm


def StoQ(state):  # interpret the state and assign location indices inside the Q matrix
    return tuple(np.round((state - Smin) * Qdimn).astype(int))


def QLearn(Q, d, w, eps, eps_min, num_games):
    wins = 0
    eps_adj = (eps - eps_min) / num_games

    for _ in tqdm(range(num_games)):
        loc = StoQ(env.reset())
        done = False
        while not done:  # automatically gets satisfied after 200 actions
            Qold = Q[loc]

            if np.random.random() < eps:
                a = env.action_space.sample()  # pick a random
            else:
                a = np.argmax(Qold)  # pick the optimal

            state, reward, done, info = env.step(a)  # execute the action

            loc_new = StoQ(state)

            won = done and not info
            # Change Q
            Qold[a] = reward if won else (1 - w) * Qold[a] + w * (reward + d * np.max(Q[loc_new]))

            loc = loc_new
        wins += won
        eps -= eps_adj

    return wins


def Qtest(Q, runs=5000):
    wins = 0
    for _ in tqdm(range(runs)):
        state = env.reset()
        done = False
        # counter = 0
        while not done:
            loc = StoQ(state)
            a = np.argmax(Q[loc])
            state, reward, done, info = env.step(a)
        wins += not info
    return wins


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
wins = QLearn(  # Trains a Q-matrix
    Q,
    w=0.15,  # learning rate
    d=0.95,  # discount rate
    eps=0.80,  # epsilon greedy strategy
    eps_min=-3,
    num_games=2000)  # number of training games

print('Training ended. Number of wins:', wins)
print('Test ended. Number of wins:', Qtest(Q, runs=500))

env.close()  # close the openai gym environment
