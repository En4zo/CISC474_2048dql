import numpy as np
import random
import tensorflow as tf
from copy import deepcopy
import game
from agent import Agent
import math
import gb
import saver

"""
The train.py contains the code for training the weight of the model
By training the weight, the score of each run will be recorded to see the change
"""

# create an Agent object here
a = Agent()
def train(episode=100):  # number of episodes
    # the following part of the program will modify the weights from each layer of the object a
    # the array to take the loss from each episode
    l_value = []

    # the array to take the score from each episode
    scores = []

    # to store final parameters
    result = {}

    mem_capacity = 3000

    # to store states and labels of the game for training
    # states of the game
    replay_memory = list()

    # labels of the states
    replay_labels = list()

    # for single example
    single_output = a.model(a.single_dataset)

    # for batch data
    logits = a.model(a.tf_batch_dataset)

    # loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(a.tf_batch_labels, logits)), axis=1)) / 2.0

    # optimizer
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.compat.v1.train.exponential_decay(float(a.alpha), global_step, 1000, 0.9, staircase=True)
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.compat.v1.Session() as session:
        tf.compat.v1.global_variables_initializer().run()
        # calculate the amount of the iterations from one
        total_iters = 1

        for it in range(episode):
            # initiate a new game each time
            board = game.new_game()
            game.add(board)
            game.add(board)

            # the game is always not over at the beginning
            finish = 'no'

            # total_score of this episode
            total_score = 0

            while finish == 'no':
                # get the required move for this state
                state = game.change_values(deepcopy(board))
                state = np.array(state, dtype=np.float32).reshape(1, 4, 4, 16)
                feed_dict = {a.single_dataset: state}
                control_scores = session.run(single_output, feed_dict=feed_dict)

                # find the move with max Q value
                control_buttons = np.flip(np.argsort(control_scores), axis=1)

                # copy the Q-values as labels
                labels = deepcopy(control_scores[0])

                prev_board = deepcopy(board)
                # store prev max
                prev_max = np.max(prev_board)

                #  less than epsilon generates random move
                if random.uniform(0, 1) < a.epsilon:
                    # find legal moves
                    finish, legal_moves = a.legal_moves(prev_board)

                    if finish == 'lose':
                        continue

                    # generate random move.
                    con = random.sample(legal_moves, 1)[0]

                    # apply the move
                    temp_state, _, score = game.controls[con](deepcopy(prev_board))
                    total_score += score
                    finish = game.check_terminal(temp_state)

                    # get number of merges
                    empty1 = game.find_empty(prev_board)
                    empty2 = game.find_empty(temp_state)

                    if finish == 'no':
                        temp_state = game.add(temp_state)

                    board = deepcopy(temp_state)

                    # get next max after applying the move
                    next_max = np.max(temp_state)

                    # reward math.log(next_max,2)*0.1 if next_max is higher than prev max
                    labels[con] = (math.log(next_max, 2))*0.1

                    if next_max == prev_max:
                        labels[con] = 0

                    # reward is also the number of merges
                    labels[con] += (empty2-empty1)

                    # get the next state max Q-value
                    temp_state = game.change_values(temp_state)
                    temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
                    feed_dict = {a.single_dataset:temp_state}
                    temp_scores = session.run(single_output,feed_dict=feed_dict)

                    max_qvalue = np.max(temp_scores)

                    # final labels add gamma*max_qvalue
                    labels[con] = (labels[con] + a.gamma * max_qvalue)

                # generate the the max predicted move
                else:
                    for con in control_buttons[0]:
                        # apply the LEGAl Move with max q_value
                        temp_state, _, score = game.controls[con](deepcopy(prev_board))

                        # if illegal move label = 0
                        if np.array_equal(prev_board, temp_state):
                            labels[con] = 0
                            continue

                        # get number of merges
                        empty1 = game.find_empty(prev_board)
                        empty2 = game.find_empty(temp_state)

                        temp_state = game.add(temp_state)
                        board = deepcopy(temp_state)
                        total_score += score

                        # pick the maximum value
                        next_max = np.max(temp_state)

                        # reward
                        labels[con] = (math.log(next_max, 2))*0.1
                        if next_max == prev_max:
                            labels[con] = 0

                        labels[con] += (empty2-empty1)

                        # get next max qvalue
                        temp_state = game.change_values(temp_state)
                        temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
                        feed_dict = {a.single_dataset: temp_state}
                        temp_scores = session.run(single_output, feed_dict=feed_dict)

                        max_qvalue = np.max(temp_scores)

                        # final labels
                        labels[con] = (labels[con] + a.gamma * max_qvalue)
                        break

                    if np.array_equal(prev_board, board):
                        finish = 'lose'

                # decrease the epsilon value
                if it > 10000 or (a.epsilon > 0.1 and total_iters % 2500 == 0):
                    a.epsilon = a.epsilon / 1.005

                # change the matrix values and store them in memory
                prev_state = game.change_values(deepcopy(prev_board))
                prev_state = np.array(prev_state, dtype=np.float32).reshape(1, 4, 4, 16)
                replay_labels.append(labels)
                replay_memory.append(prev_state)

                # back-propagation
                if len(replay_memory) >= mem_capacity:
                    back_loss = 0
                    batch_num = 0
                    z = list(zip(replay_memory, replay_labels))
                    np.random.shuffle(z)
                    np.random.shuffle(z)
                    replay_memory, replay_labels = zip(*z)

                    for i in range(0, len(replay_memory), gb.BATCH_SIZE):
                        if i + gb.BATCH_SIZE > len(replay_memory):
                            break

                        batch_data = deepcopy(replay_memory[i:i+gb.BATCH_SIZE])
                        batch_labels = deepcopy(replay_labels[i:i+gb.BATCH_SIZE])

                        batch_data = np.array(batch_data, dtype=np.float32).reshape(gb.BATCH_SIZE, 4, 4, 16)
                        batch_labels = np.array(batch_labels, dtype=np.float32).reshape(gb.BATCH_SIZE, gb.OUTPUT_UNIT)

                        feed_dict = {a.tf_batch_dataset: batch_data, a.tf_batch_labels: batch_labels}
                        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                        back_loss += l
                        batch_num += 1
                    # print("l", l)
                    back_loss /= batch_num
                    # print(back_loss)
                    l_value.append(back_loss)

                    # store the parameters in a dictionary
                    result['conv1_layer1_weights'] = session.run(a.conv1_layer1_weights)
                    result['conv1_layer2_weights'] = session.run(a.conv1_layer2_weights)
                    result['conv2_layer1_weights'] = session.run(a.conv2_layer1_weights)
                    result['conv2_layer2_weights'] = session.run(a.conv2_layer2_weights)
                    result['fc_layer1_weights'] = session.run(a.fc_layer1_weights)
                    result['fc_layer2_weights'] = session.run(a.fc_layer2_weights)
                    result['fc_layer1_biases'] = session.run(a.fc_layer1_biases)
                    result['fc_layer2_biases'] = session.run(a.fc_layer2_biases)

                    # make new memory
                    replay_memory = list()
                    replay_labels = list()

                # update the total iteration
                total_iters += 1

            scores.append(total_score)
            print("Episode {} finished with score {}, result : {} board : {}, epsilon  : {}, learning rate : {} "
                  .format(it, total_score, finish, board, a.epsilon, session.run(learning_rate)))
            # print("loss:", l_value)
            if len(l_value) > 0:
                print("Loss : {}".format(l_value[-1]))

            # print("Current Max : {}".format(max(scores)))
            print()
        print("Max : {}".format(max(scores)))
    return result, scores, l_value


result, score, l_value = train()


# save the score and loss here
saver.save(path='./weight/', name='scores', lis=score)
saver.save(path='./weight/', name='losses', lis=l_value)

# save the path

path = r'F:\refresh\474project\weight'
weights = ['conv1_layer1_weights', 'conv1_layer2_weights', 'conv2_layer1_weights', 'conv2_layer2_weights',
           'fc_layer1_weights', 'fc_layer1_biases', 'fc_layer2_weights', 'fc_layer2_biases']

for w in weights:
    flatten = result[w].reshape(-1, 1)
    file = open(path + '\\' + w +'.csv','w')
    file.write('Sno,Weight\n')
    for i in range(flatten.shape[0]):
        file.write(str(i) + ',' + str(flatten[i][0])+'\n')
    file.close()
    print(w + " done")
