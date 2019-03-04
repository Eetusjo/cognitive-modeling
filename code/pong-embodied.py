# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
#
# Ported to Python 3 by Jami Pekkanen
#
# Substantially refactored and extended by Eetu Sjöblom and Group Epsilon

import argparse
import cv2
import gym
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Concatenate, Input, Conv2D, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s: %(message)s",
                    datefmt="%d/%m/%Y %I:%M:%S %p",
                    level=logging.DEBUG)

# Script Parameters
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
    mask_indices = np.where(paddle_column != 1)[0]
    # Get paddle indices separately
    paddle_indices = np.where(paddle_column == 1)[0]
    # Return processed screen and rows to be masked
    return screen.astype(np.float), mask_indices, paddle_indices


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


def get_model(model_type="shallow_cnn", two_channel=False, ball_position=False,
              lr=0.001, resume=None):
    input_shape = (2 if two_channel else 1, 80, 80)
    if model_type == "shallow_cnn":
        input_screen = Input(shape=input_shape)
        # Feed screen through convolutional net
        x = Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same',
                          activation='relu', init='he_uniform')(input_screen)
        # Flatten CNN output
        x = Flatten()(x)

        # If feeding in ball position as well
        if ball_position:
            input_feats = Input(shape=(1,))
            # Concantenate CNN output and extra input features
            x = Concatenate()([x, input_feats])

        # First layer of MLP
        x = Dense(64, activation='relu', init='he_uniform')(x)
        # Output layers with logprobs for actions
        out = Dense(3, activation='softmax')(x)
    elif "small_cnn":
        input_screen = Input(shape=input_shape)
        # Feed screen through convolutional net
        x = Conv2D(
            filters=32, kernel_size=5, strides=3, padding='same',
            use_bias=True, activation="relu", kernel_initializer='he_uniform'
        )(input_screen)
        x = Conv2D(
            filters=32, kernel_size=9, strides=5, padding='same',
            use_bias=True, activation="relu", kernel_initializer='he_uniform'
        )(x)

        # Flatten CNN output
        x = Flatten()(x)

        # If feeding in ball position as well
        if ball_position:
            input_feats = Input(shape=(1,))
            # Concantenate CNN output and extra input features
            x = Concatenate()([x, input_feats])

        # First layer of MLP
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        # Output layers with logprobs for actions
        out = Dense(3, activation='softmax')(x)
    elif model_type == "deep_cnn":
        input_screen = Input(shape=input_shape)
        # Feed screen through convolutional net
        x = Conv2D(
            filters=32, kernel_size=3, strides=2, padding='same',
            use_bias=True, activation=None, kernel_initializer='he_uniform'
        )(input_screen)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=16, kernel_size=3, strides=2, padding='same',
            use_bias=True, activation=None, kernel_initializer='he_uniform'
        )(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=16, kernel_size=4, strides=2, padding='same',
            use_bias=True, activation=None, kernel_initializer='he_uniform'
        )(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=16, kernel_size=5, strides=2, padding='same',
            use_bias=True, activation=None, kernel_initializer='he_uniform'
        )(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        # Flatten CNN output
        x = Flatten()(x)

        # If feeding in ball position as well
        if ball_position:
            input_feats = Input(shape=(1,))
            # Concantenate CNN output and extra input features
            x = Concatenate()([x, input_feats])

        # First layer of MLP
        x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        # Output layers with logprobs for actions
        out = Dense(3, activation='softmax')(x)

    # This in case relativw ball position included as input
    if ball_position:
        model = Model(inputs=[input_screen, input_feats], outputs=out)
    else:
        model = Model(inputs=input_screen, outputs=out)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))

    if resume:
        model.load_weights(resume)

    return model


def mask_visual_field(screen, indices):
    screen[:, :, indices, :] = 0.0
    return screen


def blur_visual_field(img, paddle_indices):
    # FIXME: Inefficient
    img = img.reshape(80, 80)
    heavy = cv2.GaussianBlur(img, (27, 27), 0)
    medium = cv2.GaussianBlur(img, (19, 19), 0)
    light = cv2.GaussianBlur(img, (5, 5), 0)

    light_start = max(paddle_indices[0] - 10, 0)
    light_end = min(paddle_indices[-1] + 10, 79)

    medium_start = max(light_start - 10, 0)
    medium_end = min(light_end + 10, 79)

    blurred = heavy
    blurred[medium_start:medium_end] = medium[medium_start:medium_end]
    blurred[light_start:light_end] = light[light_start:light_end]
    blurred[paddle_indices] = img[paddle_indices]

    return blurred.reshape(1, 1, 80, 80)


def get_ball_relative(env):
    """Get ball position relative (above/below) to paddle position.

    Adapted from Jami Pekkanen.
    Source: https://github.com/jampekka/Keras-Pong/blob/master/keras_pong_2.py
    """
    # This reads the paddle and ball positions directly from the
    # atari emulator's memory. Note that these are available even when
    # the ball is invisible between rounds (when it goes all the way up).
    # This hampers the learning somewhat as the paddle tends to go to up
    # in the beginning.

    # idx = [4, 12, 21, 49, 50, 51, 54, 56, 58, 60, 64, 67, 121, 122]
    idx = [54, 49, 21, 51]
    bally, ballx, oppy, playery = env._get_ram()[idx].astype(float)/206

    if bally <= 0 or bally == playery:
        return 0
    elif bally < playery:
        return -1
    else:
        return 1


def main(args):
    # Initialize tensorboardX writer
    writer = SummaryWriter("{}/{}/".format(args.logdir, args.name))
    # Initialize gym environment
    env = gym.make("Pong-v0")
    # Number of actions allowed (NOTHING, UP, DOWN)
    n_actions = 3
    # Reset env and get first screen
    observation = env.reset()

    # Keep previous screen in memory
    prev_x = None
    # Keep previous action in memory (movement constraints)
    prev_action = None

    # Initialize lists for batches
    x_train, y_train, rewards, ball_rels = [], [], [], []
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
    episode_number = args.start_episode - 1

    # Initialize model
    model = get_model(resume=args.resume, lr=args.lr, model_type=args.model,
                      ball_position=args.relative_vision,
                      two_channel=args.two_channel_vision)

    # Print model summary
    model.summary(print_fn=logging.info)
    # Ensure saving and logging dirs
    for dr in [args.savedir, args.logdir]:
        if not os.path.isdir(dr):
            os.mkdir(dr)

    # Begin training
    while True:
        if render:
            env.render()
        # Preprocess, consider the frame difference as features
        cur_x, mask_indices, paddle_indices \
            = pong_preprocess_screen(observation)
        # Reshape to (batch, channels, height, width)
        cur_x = cur_x.reshape((1, 1, 80, 80))
        x = cur_x - prev_x if prev_x is not None \
            else np.zeros((1, 1, 80, 80))

        # Mask parts of screen if simulating visual field. Reshape x to
        # (80, 80) because pong_preprocess_screen returns flattened array
        if args.masked_vision or args.visual_field:
            x = mask_visual_field(x, mask_indices)
        elif args.blurred_vision:
            x = blur_visual_field(x, paddle_indices)
        elif args.relative_vision:
            x = mask_visual_field(x, mask_indices)
            ball_relative = get_ball_relative(env.unwrapped)
            ball_rels.append(ball_relative)
        elif args.two_channel_vision:
            x = np.concatenate([x, cur_x], axis=1)

        # Set current (non-masked) screen as previous
        prev_x = cur_x
        # Append (masked) screen to batch
        x_train.append(x)

        # Predict probabilities for actions using the model
        aprob = model.predict(
            [x, np.array([ball_relative])] if args.relative_vision else x,
            batch_size=1
        ).flatten()
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
                if args.relative_vision:
                    model.train_on_batch(
                        [np.vstack(x_train), np.vstack(ball_rels)],
                        np.vstack(y_train),
                        sample_weight=discounted
                    )
                else:
                    model.train_on_batch(np.vstack(x_train),
                                         np.vstack(y_train),
                                         sample_weight=discounted)
                # Clear the batch
                x_train, y_train, ball_rels = [], [], []
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
    parser.add_argument("-m", "--model", default="shallow_cnn",
                        choices=["shallow_cnn", "deep_cnn", "small_cnn"],
                        help="Name for experiment. Used for saving model.")

    visual_group = parser.add_mutually_exclusive_group(required=False)
    visual_group.add_argument("--visual_field", action="store_true",
                              help="See 'masked_vision' (DEPRECATED)")
    visual_group.add_argument("--masked_vision", action="store_true",
                              help="Use masked field of vision")
    visual_group.add_argument("--blurred_vision", action="store_true",
                              help="Use blurred field of vision")
    visual_group.add_argument("--relative_vision", action="store_true",
                              help="Use masked field of vision with relative "
                                   "position of ball")
    visual_group.add_argument("--two_channel_vision", action="store_true",
                              help="Use two-channel vision")

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
