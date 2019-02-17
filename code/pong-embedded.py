# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
#
# Ported to Python 3 by Jami Pekkanen
# Tested with the theano backend (tensorflow has issues with Python 3.7)
import os
import gym
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s: %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    level=logging.DEBUG)

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
    # Normalize
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r.ravel()


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
    screen[indices, :] = 0
    return screen.ravel()


def main(args):
    # Initialize tensorboardX writer
    writer = SummaryWriter("{}/{}/".format(args.logdir, args.name))
    # Initialize gym environment
    env = gym.make("Pong-v0")
    # Number of actions allowed (NOTHING, UP. DOWN)
    n_actions = 3
    # Reset env and get first screen
    observation = env.reset()

    # Keep previous screen in memory
    prev_x = None
    # Keep previous action in memory (movement constraints)
    prev_action = None

    # Initialize lists for batches
    x_train, y_train, rewards = [], [], []
    # 'action_rewards' contains rewards for same/different action
    # 'no_movement_rewards' contains rewards for choosing action 1 vs 2,3
    action_rewards, no_movement_rewards = [], []

    # Alpha parameters if using step alpha
    if args.step_alpha:
        alpha, alpha_end, alpha_step, alpha_inter = args.step_alpha

    # For smoothing per-episode reward logging
    reward_history = deque(maxlen=100)
    # Keep track of rewards in episode
    reward_sum = 0
    # Keep track of episode
    episode_number = args.start_episode

    # Initialize model
    model = learning_model(
        num_actions=n_actions, resume=args.resume, lr=args.lr
    )

    # Begin training
    while True:
        if render:
            env.render()
        # Preprocess, consider the frame difference as features
        cur_x, mask_indices = pong_preprocess_screen(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)

        # Mask parts of screen if simulating visual field. Reshape x to
        # (80, 80) because pong_preprocess_screen returns flattened array
        x = mask_visual_field(x.reshape(80, 80), mask_indices) \
            if args.visual_field else x

        # Set current (non-masked) screen as previous
        prev_x = cur_x
        # Append (masked) screen to batch
        x_train.append(x)

        # Predict probabilities for actions using the model
        aprob = model.predict(x.reshape([1, -1]), batch_size=1).flatten()
        # Sample action. Original samples from [0, 5], but {0, 1, 4, 5} do
        # nothing. Sampling from [1, 3] allows same actions (nothing, up, down)
        # with less computation.
        action = np.random.choice([1, 2, 3], 1, p=aprob)[0]

        # +1 if took same actions as last step, -1 if different
        # FIXME: Should we take into account when no action was taken?
        if action == prev_action or prev_action is None:
            action_rewards.append(1.)
        else:
            action_rewards.append(-1.)

        if action == 1:
            no_movement_rewards.append(1.)
        else:
            no_movement_rewards.append(-1.)

        # Store action as one-hot encoded vector
        y = np.zeros([n_actions])
        # -1 compensates for sampling from [1, 3]
        y[action - 1] = 1
        # Gather action for training batch
        y_train.append(y)

        # Take step in game
        observation, reward, done, info = env.step(action)
        # Gather sum for this episode
        reward_sum += reward
        # Gather rewards for later
        rewards.append(reward)

        # Game finished
        if done:
            episode_number += 1
            # Periodically update the model
            if episode_number % update_frequency == 0:
                # Calculate weights for rewards ("discount")
                discounted = discount_rewards(np.vstack(rewards), args.gamma)

                # Use movement contraings if alpha set and  > 0
                if (args.alpha and args.alpha > 0):
                    action_rewards = np.array(action_rewards, dtype=np.float)
                    # Scale movement constraint weights by alpha
                    discounted = np.add(discounted, args.alpha*action_rewards)
                elif args.step_alpha:
                    action_rewards = np.array(action_rewards, dtype=np.float)
                    # Scale movement constraint weights by alpha
                    discounted = np.add(discounted, alpha*action_rewards)

                # Penalize movement and encourage no-action if beta > 0
                if args.beta > 0:
                    no_movement_rewards = np.array(
                        no_movement_rewards, dtype=np.float
                    )
                    discounted = np.add(
                        discounted, args.beta*no_movement_rewards
                    )

                # Reset rewards list
                rewards = []

                # Train on this batch, weighting samples by discounted rewards
                model.train_on_batch(np.squeeze(np.vstack(x_train)),
                                     np.vstack(y_train),
                                     sample_weight=discounted)
                # Clear the batch
                x_train, y_train = [], []
                # Save a checkpoint of the model
                path = '{}/{}.h5'.format(args.savedir, args.name)
                # Remove old model checkpoint
                os.remove(path) if os.path.exists(path) else None
                # Save model
                model.save_weights(path)

            reward_history.append(reward_sum)
            # Reset environment and print the results for this episode
            logging.info(
                "Episode: %d. " % episode_number +
                "Total Episode Reward: %f. " % reward_sum +
                "Running Mean: %f" % (sum(reward_history)/len(reward_history))
            )

            # Update alpha if we are at interval and haven't reached max
            # (and using step_alpha)
            if args.step_alpha and episode_number % alpha_inter == 0 \
                    and alpha < alpha_end:
                alpha += alpha_step
                if alpha < alpha_end:
                    logging.info("Updating alpha-value at interval. "
                                 "New alpha: {}".format(alpha))
                else:
                    logging.info("Updating alpha-value at interval. "
                                 "Reached max alpha defined by user. "
                                 "Final alpha: {}".format(alpha))

            # Log rewards for this episode using tensorboardX
            writer.add_scalar("reward", reward_sum, episode_number)

            reward_sum = 0
            # Reset game environment and get initial observation
            observation = env.reset()
            # Set last screen and action to None
            prev_x = None
            prev_action = None
            action_rewards = []
            no_movement_rewards = []

        # Optionally log per-round stats
        if reward != 0 and args.verbose:
            logging.info(('Episode %d Result: ' % episode_number)
                         + ('Defeat!' if reward == -1 else 'VICTORY!'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="pong",
                        help="Name for experiment. Used for saving model.")

    alpha_group = parser.add_mutually_exclusive_group(required=False)
    alpha_group.add_argument("--alpha", type=float, default=None,
                             help="Strength of movement constraints")
    alpha_group.add_argument("--step_alpha", type=float, nargs=4,
                             default=None,
                             help="See alpha. Controls alpha parameter in a "
                                  "step-wise manner. list of floats: "
                                  "[start, end, step, interval]")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Parameter for encouraging no movement")
    parser.add_argument("--gamma", type=float, default=0.99, required=False,
                        help="Parameter for discounting rewards")
    parser.add_argument("--lr", type=float, default=0.001, required=False,
                        help="Learning rate for training.")
    parser.add_argument("--visual_field", action="store_true",
                        help="Use a restricted visual field for paddle")
    # Args related to training, saving and logging
    parser.add_argument("-s", "--savedir", default="./saved/",
                        help="Directory for saving models")
    parser.add_argument("-l", "--logdir", default="./log/",
                        help="Directory for logging model training")
    parser.add_argument("-r", "--resume", default=None,
                        help="Path of pre-trained model")
    parser.add_argument("--start_episode", default=0, type=int,
                        help="Start training from this episode. Quick fix for "
                             "not saving proper checkpoint. Change?")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging to stdout")
    args = parser.parse_args()
    main(args)
