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
from tensorboardX import SummaryWriter

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D

# Script Parameters
# FIXME: Eventually move all of this somewhere else
input_dim = 80 * 80
update_frequency = 1
render = False


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


def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    return discounted_r


def learning_model(input_dim=80*80, num_actions=3,
                   model_type="shallow_cnn", lr=0.001, resume=None):
    model = Sequential()
    if model_type == "mlp":
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(num_actions, activation='softmax'))
        opt = RMSprop(lr=lr)
    elif model_type == "shallow_cnn":
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu', init='he_uniform'))
        model.add(Dense(num_actions, activation='softmax'))
        opt = Adam(lr=lr)
    elif model_type == "deep_cnn":
        raise NotImplementedError("Deep CNN model not implemented.")

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume:
        model.load_weights(resume)

    return model


def mask_visual_field(screen, indices):
    # If game ongoing, mask visual field
    if len(indices) > 0:
        screen[indices, :] = 0
    return screen.ravel()


def main(args):
    # Initialize tensorboardX writer
    writer = SummaryWriter("{}/{}/".format(args.logdir, args.name))

    # Initialize gym environment
    env = gym.make("Pong-v0")
    # Number of actions allowed
    n_actions = 3
    # Reset env and get first screen
    observation = env.reset()
    # Keep previous screen in memory
    prev_x = None
    x_train, dlogps, rewards, probs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    # Initialize model
    model = learning_model(num_actions=n_actions, resume=args.resume)

    # Begin training
    while True:
        if render:
            env.render()
        # Preprocess, consider the frame difference as features
        cur_x, mask_indices = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
        x = mask_visual_field(x.reshape(80, 80), mask_indices) \
            if args.visual_field else x
        prev_x = cur_x
        x_train.append(x)

        # Predict probabilities from the Keras model
        aprob = model.predict(x.reshape([1, -1]), batch_size=1).flatten()
        probs.append(aprob)

        # Sample action
        # Original samples from [0, 5], but {0, 1, 4, 5} do nothing. Sampling
        # from [1, 3] allows same actions (nothing, up, down) with less
        # computation.
        action = np.random.choice([1, 2, 3], 1, p=aprob)[0]

        # Store action as one-hot encoded vector
        y = np.zeros([n_actions])
        # -1 compensates for sampling from [1, 3]
        y[action - 1] = 1
        # Append features and labels for the episode-batch
        dlogps.append(y.astype('float32') - aprob)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)
        if done:
            episode_number += 1

            epdlogp = np.vstack(dlogps)
            rewards = np.vstack(rewards)
            discounted = discount_rewards(np.vstack(rewards), args.gamma)
            epdlogp *= discounted

            # train_X contains the screen inputs
            dlogps, rewards = [], []

            # Periodically update the model
            if episode_number % update_frequency == 0:
                y_train = probs + args.lr * epdlogp

                # print('Training Snapshot:')
                # print(y_train)

                model.train_on_batch(np.squeeze(np.vstack(x_train)), y_train)

                # Clear the batch
                x_train = []
                probs = []

                # Save a checkpoint of the model
                path = '{}/{}.h5'.format(args.savedir, args.name)
                # Remove old model ceckpoint
                os.remove(path) if os.path.exists(path) else None
                model.save_weights(path)

            # Reset the current environment and print the current results
            running_reward = reward_sum if running_reward is None \
                else running_reward * 0.99 + reward_sum * 0.01
            print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward))

            writer.add_scalar("reward", reward_sum, episode_number)

            reward_sum = 0
            observation = env.reset()
            prev_x = None
        if reward != 0 and args.verbose:
            print(('Episode %d Result: ' % episode_number)
                  + ('Defeat!' if reward == -1 else 'VICTORY!'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="pong",
                        help="Name for experiment. Used for saving model.")
    # Model definition
    parser.add_argument("--gamma", type=float, default=0.99, required=False,
                        help="Gamma parameter for discounting rewards")
    parser.add_argument("--lr", type=float, default=0.001, required=False,
                        help="Learning rate for training.")
    parser.add_argument("--visual_field", action="store_true",
                        help="Use a restricted visual field for paddle.")
    # Args related to training, saving and logging
    parser.add_argument("-s", "--savedir", default="./saved/",
                        help="Directory for saving models")
    parser.add_argument("-l", "--logdir", default="./log/",
                        help="Directory for logging model training.")
    parser.add_argument("-r", "--resume", default=None,
                        help="Path of pre-trained model.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging to stdout.")
    args = parser.parse_args()
    main(args)
