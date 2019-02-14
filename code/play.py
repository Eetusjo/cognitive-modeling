# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
#
# Ported to Python 3 by Jami Pekkanen
# Tested with the theano backend (tensorflow has issues with Python 3.7)
import os
import gym
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D

# Script Parameters
input_dim = 80 * 80
learning_rate = 0.001
render = True


def pong_preprocess_screen(screen):
    screen = screen[35:195]
    screen = screen[::2, ::2, 0]
    screen[screen == 236] = 1
    screen[screen == 213] = 1
    screen[screen == 92] = 1
    screen[screen < 1] = 0
    screen[screen > 1] = 0

    # Get the correct column to retrieve paddle position
    paddle_column = screen[:, 71]
    # Get correct rows (paddle position)
    indices = np.where(paddle_column != 1)[0]
    # Return processed screen and paddle position as row indices
    return screen.astype(np.float).ravel(), indices


def learning_model(input_dim=80*80, num_actions=2, model_type=1, resume=None):
    model = Sequential()
    if model_type == 0:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(num_actions, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    else:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu', init='he_uniform'))
        model.add(Dense(num_actions, activation='softmax'))
        opt = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt)

    if resume:
        model.load_weights(resume)

    return model


def mask_visual_field(screen, indices):
    # If game ongoing, mask visual field
    if len(indices) > 0:
        screen[indices] = 0
    return screen


def main(args):
    # Initialize gym environment
    env = gym.make("Pong-v0")
    # Get number of actions available in environment
    number_of_inputs = env.action_space.n # This is incorrect for Pong (but whatever)
    # Reset env and get first screen
    observation = env.reset()
    # Keep previous screen in memory
    prev_x = None

    # Initialize model
    model = learning_model(num_actions=number_of_inputs, resume=args.resume)

    # Begin training
    while True:
        if render:
            env.render()
            time.sleep(0.01)

        # Preprocess, consider the frame difference as features
        cur_x, paddle_indices = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        if args.visual_field:
            x = mask_visual_field(x, paddle_indices)
        prev_x = cur_x

        # Predict probabilities from the Keras model
        aprob = model.predict(x.reshape([1, -1]), batch_size=1).flatten()
        # Sample action
        aprob = aprob/np.sum(aprob)
        action = np.random.choice(number_of_inputs, 1, p=aprob)[0]

        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            prev_x = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", default=None,
                        help="Path of pre-trained model.")
    parser.add_argument("--visual_field", action="store_true",
                        help="Use a restricted visual field for paddle.")
    args = parser.parse_args()
    main(args)
