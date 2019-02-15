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
    # Get rid of upper and lower margins
    screen = screen[35:195]
    # Downsample to 80x80
    screen = screen[::2, ::2, 0]
    # Change backgrounds to 0
    screen[screen == 144] = 0
    screen[screen == 109] = 0
    # Change paddles and ball to 1
    screen[screen != 0] = 1

    # Get the correct column to retrieve paddle position
    paddle_column = screen[:, 71]
    # Get correct rows for maskig (exclude paddle position)
    indices = np.where(paddle_column != 1)[0]
    # Return processed screen and rows to be masked
    return screen.astype(np.float).ravel(), indices


def learning_model(input_dim=80*80, num_actions=3, model_type=1, resume=None):
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
    # Number of actions allowed
    n_actions = 3
    # Reset env and get first screen
    observation = env.reset()
    # Keep previous screen in memory
    prev_x = None
    x_train, rewards = [], []
    # Keep track of rewards in episode
    reward_sum = 0

    # Initialize models log
    model = learning_model(
        num_actions=n_actions, resume=args.resume,
    )

    while True:
        env.render()
        time.sleep(0.05)
        # Preprocess, consider the frame difference as features
        cur_x, mask_indices = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        x = mask_visual_field(x.reshape(80, 80), mask_indices) \
            if args.visual_field else x
        prev_x = cur_x
        x_train.append(x)

        # Predict probabilities from the Keras model
        aprob = model.predict(x.reshape([1, -1]), batch_size=1).flatten()

        # Sample action
        # Original samples from [0, 5], but {0, 1, 4, 5} do nothing. Sampling
        # from [1, 3] allows same actions (nothing, up, down) with less
        # computation.
        action = np.random.choice([1, 2, 3], 1, p=aprob)[0]

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)
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
