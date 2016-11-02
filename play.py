"""Playing script for the network."""

from __future__ import print_function

import os
import fnmatch
import numpy as np
import tensorflow as tf

import chess.pgn

LABELS_DIRECTORY = './labels'
IMAGE_SIZE = 8
FEATURE_PLANES = 8
LABEL_SIZE = 6100
FILTERS = 128
HIDDEN = 512

labels = []


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def model(data):
    # network weights
    W_conv1 = weight_variable([IMAGE_SIZE, IMAGE_SIZE, FEATURE_PLANES, FILTERS])
    b_conv1 = bias_variable([FILTERS])

    W_conv2 = weight_variable([5, 5, FILTERS, FILTERS])
    b_conv2 = bias_variable([FILTERS])

    W_conv3 = weight_variable([3, 3, FILTERS, FILTERS])
    b_conv3 = bias_variable([FILTERS])

    W_fc1 = weight_variable([HIDDEN, HIDDEN])
    b_fc1 = bias_variable([HIDDEN])

    W_fc2 = weight_variable([HIDDEN, LABEL_SIZE])
    b_fc2 = bias_variable([LABEL_SIZE])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1, 1) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 3) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_flat = tf.reshape(h_pool3, [-1, HIDDEN])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    return readout


# Input data.
tf_prediction = tf.placeholder(tf.float32,
                                  shape=(None,
                                  IMAGE_SIZE,
                                  IMAGE_SIZE,
                                  FEATURE_PLANES))

# Training computation.
logits = model(tf_prediction)

# Predictions for the model.
train_prediction = tf.nn.softmax(logits)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("logdir")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print ("Successfully loaded:", checkpoint.model_checkpoint_path)


def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def read_labels(directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    labels_array = []
    for filename in files:
        with open(filename) as f:
            lines = str(f.readlines()[0]).split(" ")
            for label in lines:
                if(label != " " and label != '\n'):
                    labels_array.append(label)
    return labels_array


def replace_tags(board):
    board_san = board.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    for i in range(len(board.split(" "))):
        if i > 0 and board.split(" ")[i] != '':
            board_san += " " + board.split(" ")[i]
    return board_san


def reformat(game):
    board_state = replace_tags(game.replace("/", ""))
    # All pieces plane
    board_pieces = list(board_state.split(" ")[0])
    board_pieces = [ord(val) for val in board_pieces]
    board_pieces = np.reshape(board_pieces, (IMAGE_SIZE, IMAGE_SIZE))
    # Only spaces plane
    board_blank = [int(val == '1') for val in board_state.split(" ")[0]]
    board_blank = np.reshape(board_blank, (IMAGE_SIZE, IMAGE_SIZE))
    # Only white plane
    board_white = [int(val.isupper()) for val in board_state.split(" ")[0]]
    board_white = np.reshape(board_white, (IMAGE_SIZE, IMAGE_SIZE))
    # Only black plane
    board_black = [int(not val.isupper() and val != '1') for val in board_state.split(" ")[0]]
    board_black = np.reshape(board_black, (IMAGE_SIZE, IMAGE_SIZE))
    # One-hot integer plane current player turn
    current_player = board_state.split(" ")[1]
    current_player = np.full((IMAGE_SIZE, IMAGE_SIZE), int(current_player == 'w'), dtype=int)
    # One-hot integer plane extra data
    extra = board_state.split(" ")[4]
    extra = np.full((IMAGE_SIZE, IMAGE_SIZE), int(extra), dtype=int)
    # One-hot integer plane move number
    move_number = board_state.split(" ")[5]
    move_number = np.full((IMAGE_SIZE, IMAGE_SIZE), int(move_number), dtype=int)
    # Zeros plane
    zeros = np.full((IMAGE_SIZE, IMAGE_SIZE), 0, dtype=int)

    planes = np.vstack((np.copy(board_pieces),
                        np.copy(board_white),
                        np.copy(board_black),
                        np.copy(board_blank),
                        np.copy(current_player),
                        np.copy(extra),
                        np.copy(move_number),
                        np.copy(zeros)))
    planes = np.reshape(planes, (1, IMAGE_SIZE, IMAGE_SIZE, FEATURE_PLANES))
    return planes


def main():
    labels = read_labels(LABELS_DIRECTORY, "*.txt")
    print('\nPlaying...\nComputer plays white.\n')
    board = chess.Board()
    while(not board.is_game_over()):
        # We get the movement prediction
        game_state = reformat(board.fen())
        feed_dict = {tf_prediction: game_state}
        predictions = sess.run([train_prediction], feed_dict=feed_dict)
        legal_moves = []
        for move in board.legal_moves:
            legal_moves.append(board.san(move))
        legal_labels = [int(mov in legal_moves) for mov in labels]
        movement = labels[np.argmax(predictions[0] * legal_labels)]
        print('The computer wants to move to:', movement)
        try:
            if(board.parse_san(movement) in board.legal_moves):
                print ("and it's a valid movement :).")
                board.push_san(movement)
                print("\n")
                print(board)
                print("\n")
            else:
                print("but that is NOT a valid movement.")
        except:
            print ("but that is NOT a valid movement :(.")

        # we move now
        moved = False
        while not moved:
            try:
                movement = raw_input('Enter your movement: ')
                if(board.parse_san(movement) in board.legal_moves):
                    print ("That is a valid movement.")
                    board.push_san(movement)
                    print("\n")
                    print(board)
                    print("\n")
                    moved = True
                else:
                    print ("That is NOT a valid movement :(.")
            except:
                print ("but that is NOT a valid movement :(.")
    print("\nEnd of the game.")
    print("Game result:")
    print(board.result())


if __name__ == '__main__':
    main()
