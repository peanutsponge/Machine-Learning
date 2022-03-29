from ludo import make
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

## does one-hot encoding for us
from keras.utils import np_utils

## load sequential models
from keras.models import Sequential

## load a dense layer all-to-all connection
from keras.layers import Dense

Nin = 17
N1 = 300
N2 = 100
Nout = 4
def CreateModel():
    ## create model instance
    model = Sequential()
    ## call add memeber function of model to append some layer
    model.add(Dense(N1, input_dim=Nin, kernel_initializer='normal',
                    activation="relu"))
    ## add first hidden layer always match input and dimension of actual
    ## and previous layer
    model.add(Dense(N2, input_dim=N1,
                    kernel_initializer="normal", activation="relu"))
    ## add the output layer, since we are doing a classification problem
    ## the number of output layers has to match the number of classes
    ## since we have 10 digits 0..9 ->
    model.add(Dense(Nout, input_dim=N2,
                    kernel_initializer="normal", activation="softmax"))
    ## and last keras allows us to compile the model to get all the compute power
    ## available, and we can define a optimizer and a loss function
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=['accuracy'])
    return model


# create an instance of the game with 4 players
env = make(num_players=4)

# reset the game
obs, rew, done, info = env.reset()

while True:
    # get an action from the random player
    action = 1

    # pass the action and get the new gamestate
    obs, rew, done, info = env.step(action)

    # render for graphical representation of gamestate
    env.render()

    # quit if game is finished
    if done:
        break