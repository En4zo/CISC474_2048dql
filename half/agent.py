import tensorflow.compat.v1 as tf
import numpy as np
from copy import deepcopy
import random
import game
import gb
tf.disable_v2_behavior()


class Agent:
    def __init__(self, alpha=1e3, gamma=0.9, epsilon=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.tf_batch_dataset, self.tf_batch_labels, self.single_dataset = self.dataset()
        self.conv1_layer1_weights, self.conv2_layer1_weights = self.conv_layer1()
        self.conv1_layer2_weights, self.conv2_layer2_weights = self.conv_layer2()

        self.fc_layer1_weights, self.fc_layer1_biases, self.fc_layer2_weights, self.fc_layer2_biases = self.fully_connected()

    @staticmethod
    def dataset():
        return tf.placeholder(tf.float32, shape=(gb.BATCH_SIZE, 4, 4, 16)), \
               tf.placeholder(tf.float32, shape=(gb.BATCH_SIZE, gb.OUTPUT_UNIT)), \
               tf.placeholder(tf.float32, shape=(1, 4, 4, 16))

    # convolutional layers
    # the weights fot layer 1
    @staticmethod
    def conv_layer1():
        return tf.Variable(tf.truncated_normal([1, 2, gb.INPUT_UNIT, gb.DEPTH1], mean=0, stddev=0.01)), \
               tf.Variable(tf.truncated_normal([2, 1, gb.INPUT_UNIT, gb.DEPTH1], mean=0, stddev=0.01))

    # the weights for layer 2
    @staticmethod
    def conv_layer2():
        return tf.Variable(tf.truncated_normal([1, 2, gb.DEPTH1, gb.DEPTH2], mean=0, stddev=0.01)), \
               tf.Variable(tf.truncated_normal([2, 1, gb.DEPTH1, gb.DEPTH2], mean=0, stddev=0.01))

    # the fully connected layer
    @staticmethod
    def fully_connected():
        return tf.Variable(tf.truncated_normal([gb.EXPAND_SIZE, gb.HIDDEN_UNIT], mean=0, stddev=0.01)), \
               tf.Variable(tf.truncated_normal([1, gb.HIDDEN_UNIT], mean=0, stddev=0.01)), \
               tf.Variable(tf.truncated_normal([gb.HIDDEN_UNIT, gb.OUTPUT_UNIT], mean=0, stddev=0.01)), \
               tf.Variable(tf.truncated_normal([1, gb.OUTPUT_UNIT], mean=0, stddev=0.01))

    # find legal moves from the board
    @staticmethod
    def legal_moves(board):
        legal_moves = list()
        for i in range(4):
            next_board, _, _ = game.controls[i](deepcopy(board))
            if not np.array_equal(next_board, board):
                legal_moves.append(i)
        if len(legal_moves) == 0:
            return 'lose', legal_moves
        return 'not over', legal_moves

    # perform a random move
    @staticmethod
    def random_move(next_board, legal_moves):
        move = random.sample(legal_moves, 1)[0]
        next_board, score = game.controls[move](next_board)
        done = game.check_terminal(next_board)
        return done, move, next_board, score

    # check the number of merges
    @staticmethod
    def check_merges(current_board, next_board):
        return game.findemptyCell(next_board) - game.findemptyCell(current_board)

    # update q value
    @staticmethod
    def update_label(labels, prev_max, next_max, merges, move):
        labels[move] = next_max * 0.1
        if next_max == prev_max:
            labels[move] = 0
        labels[move] += merges
        return labels

    # model
    def model(self, dataset):
        # layer1
        conv1 = tf.nn.conv2d(dataset, self.conv1_layer1_weights,[1,1,1,1],padding='VALID')
        conv2 = tf.nn.conv2d(dataset, self.conv2_layer1_weights,[1,1,1,1],padding='VALID')

        # layer1 relu activation
        relu1 = tf.nn.relu(conv1)
        relu2 = tf.nn.relu(conv2)

        # layer2
        conv11 = tf.nn.conv2d(relu1, self.conv1_layer2_weights,[1,1,1,1],padding='VALID')
        conv12 = tf.nn.conv2d(relu1, self.conv2_layer2_weights,[1,1,1,1],padding='VALID')

        conv21 = tf.nn.conv2d(relu2, self.conv1_layer2_weights,[1,1,1,1],padding='VALID')
        conv22 = tf.nn.conv2d(relu2, self.conv2_layer2_weights,[1,1,1,1],padding='VALID')

        # layer2 relu activation
        relu11 = tf.nn.relu(conv11)
        relu12 = tf.nn.relu(conv12)
        relu21 = tf.nn.relu(conv21)
        relu22 = tf.nn.relu(conv22)

        # get shapes of all activations
        shape1 = relu1.get_shape().as_list()
        shape2 = relu2.get_shape().as_list()

        shape11 = relu11.get_shape().as_list()
        shape12 = relu12.get_shape().as_list()
        shape21 = relu21.get_shape().as_list()
        shape22 = relu22.get_shape().as_list()

        # put all the conv layers into one set
        hidden1 = tf.reshape(relu1, [shape1[0], shape1[1]*shape1[2]*shape1[3]])
        hidden2 = tf.reshape(relu2, [shape2[0], shape2[1]*shape2[2]*shape2[3]])

        hidden11 = tf.reshape(relu11, [shape11[0], shape11[1]*shape11[2]*shape11[3]])
        hidden12 = tf.reshape(relu12, [shape12[0], shape12[1]*shape12[2]*shape12[3]])
        hidden21 = tf.reshape(relu21, [shape21[0], shape21[1]*shape21[2]*shape21[3]])
        hidden22 = tf.reshape(relu22, [shape22[0], shape22[1]*shape22[2]*shape22[3]])

        # concatenation
        hidden = tf.concat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], axis=1)

        # full connected layers
        hidden = tf.matmul(hidden, self.fc_layer1_weights) + self.fc_layer1_biases
        hidden = tf.nn.relu(hidden)

        # output layer
        output = tf.matmul(hidden, self.fc_layer2_weights) + self.fc_layer2_biases

        # return output
        return output




