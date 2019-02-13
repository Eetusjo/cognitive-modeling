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

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D

# Script Parameters
input_dim = 80 * 80
update_frequency = 1
learning_rate = 0.001
resume = False
render = False


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

    # If game in ongoing, mask screen outside visual field
    if len(indices) > 0:
        screen[indices] = 2

    return screen.astype(np.float).ravel()


def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def learning_model(input_dim=80*80, num_actions=2, model_type=1):
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
        model.load_weights('pong_model_checkpoint.h5')

    return model


def main(args):
    # Initialize gym environment
    env = gym.make("Pong-v0")
    # Get number of actions available in environment
    number_of_inputs = env.action_space.n  # This is incorrect for Pong (but whatever)
    # Reset env and get first screen
    observation = env.reset()
    # Keep previous screen in memory
    prev_x = None
    xs, dlogps, drs, probs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    train_X = []
    train_y = []

    # Initialize model
    model = learning_model(num_actions=number_of_inputs)

    # Begin training
    while True:
        if render:
            env.render()
        # Preprocess, consider the frame difference as features
        cur_x = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        prev_x = cur_x
        xs.append(x)

        # Predict probabilities from the Keras model
        aprob = model.predict(x.reshape([1, -1]), batch_size=1).flatten()
        probs.append(aprob)

        # Sample action
        aprob = aprob/np.sum(aprob)
        action = np.random.choice(number_of_inputs, 1, p=aprob)[0]

        y = np.zeros([number_of_inputs])
        y[action] = 1
        # Append features and labels for the episode-batch
        dlogps.append(np.array(y).astype('float32') - aprob)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        if done:
            episode_number += 1
            # epx = np.vstack(xs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            discounted_epr = discount_rewards(epr, args.gamma)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr

            # Slowly prepare the training batch
            train_X.append(xs)
            train_y.append(epdlogp)
            xs, dlogps, drs = [], [], []
            # Periodically update the model
            if episode_number % update_frequency == 0:
                y_train = probs + learning_rate * np.squeeze(np.vstack(train_y)) # Hacky WIP

                print('Training Snapshot:')
                print(y_train)
                model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)

                # Clear the batch
                train_X = []
                train_y = []
                probs = []

                # Save a checkpoint of the model
                os.remove('pong_model_checkpoint.h5') \
                    if os.path.exists('pong_model_checkpoint.h5') else None
                model.save_weights('pong_model_checkpoint.h5')

            # Reset the current environment nad print the current results
            running_reward = reward_sum if running_reward is None \
                else running_reward * 0.99 + reward_sum * 0.01
            print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()
            prev_x = None
        if reward != 0:
            print(('Episode %d Result: ' % episode_number)
                  + ('Defeat!' if reward == -1 else 'VICTORY!'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gamma", type=float, default=0.99,
                        help="Gamma parameter for discounting rewards")
    args = parser.parse_args()
    main(args)
