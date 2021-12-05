import numpy as np
import random
import tensorflow as tf
from copy import deepcopy
import game
from agent import Agent
import time


BATCH_SIZE = 512
OUTPUT_UNIT = 4


def train(episodes=20000, max_replay=2000):
    a = Agent()
    scores = []
    losses = []
    logs = []
    outcomes = {}

    # to store states and lables of the game for training states
    replay_memory = list()

    # labels of the states
    replay_labels = list()

    # for single example
    single_output = a.model(a.single_dataset)

    # for batch data
    logits = a.model(a.tf_batch_dataset)

    # loss
    loss = tf.square(tf.subtract(a.tf_batch_labels,logits))
    loss = tf.reduce_sum(loss,axis=1) #keep_dims=True
    loss = tf.reduce_mean(loss)/2.0

    # optimizer
    global_step = tf.Variable(0)  # count the number of steps taken.
    temp_lr = tf.compat.v1.train.exponential_decay(float(a.alpha), global_step, 1000, 0.90, staircase=True)
    optimizer = tf.compat.v1.train.RMSPropOptimizer(temp_lr).minimize(loss, global_step=global_step)

    with tf.compat.v1.Session() as session:

        tf.compat.v1.global_variables_initializer().run()

        iterations = 1

        start_time = time.time()
        for e in range(episodes):
            # initilize game
            board = game.new_game(4)
            game.randomfill(board)
            game.randomfill(board)

            done = 'not over'
            total_score = 0

            # play the game until it's over
            while(done=='not over'):
                current_board = deepcopy(board)

                # get the required move for this state
                current_state = deepcopy(board)
                current_state = game.change_values(current_state)
                current_state = np.array(current_state,dtype = np.float32).reshape(1,4,4,16)
                feed_dict = {a.single_dataset:current_state}
                control_scores = session.run(single_output,feed_dict=feed_dict)

                # find the move with max Q value
                control_buttons = np.flip(np.argsort(control_scores),axis=1)

                # copy q values
                labels = deepcopy(control_scores[0])
                prev_max = np.max(current_board)

                # generate random move
                if(random.uniform(0,1) < a.epsilon):
                    # find legal moves
                    done, legal_moves = a.find_legal_moves(current_board)
                    if done == 'lose': continue

                    # apply a random move
                    next_board = deepcopy(current_board)
                    done, move, next_board, score = a.make_random_move(next_board, legal_moves)
                    total_score += score
                    merges = a.check_merges(current_board, next_board)

                    if done == 'not over':
                        next_board = game.randomfill(next_board)

                    board = deepcopy(next_board)

                    # collect rewards
                    next_max = np.max(next_board)
                    labels = a.update_label(labels, prev_max, next_max, merges, move)

                    # max(Q)
                    next_board = game.change_values(next_board)
                    next_board = np.array(next_board,dtype = np.float32).reshape(1,4,4,16)
                    feed_dict = {a.single_dataset:next_board}
                    temp_scores = session.run(single_output,feed_dict=feed_dict)

                    max_qvalue = np.max(temp_scores)

                    # update q value
                    labels[move] = (labels[move] + a.gamma*max_qvalue)

                # greedy move based on max(Q)
                else:
                    for con in control_buttons[0]:
                        prev_state = deepcopy(current_board)

                        # apply the LEGAl Move with max q_value
                        next_board,score = game.controls[con](prev_state)

                        #if illegal move label = 0
                        if(np.array_equal(current_board,next_board)):
                            labels[con] = 0
                            continue

                        merges = a.check_merges(current_board, next_board)

                        next_board = game.randomfill(next_board)
                        board = deepcopy(next_board)
                        total_score += score

                        # collect rewards
                        next_max = np.max(next_board)
                        labels = a.update_label(labels, prev_max, next_max, merges, con)

                        # get next max q value
                        next_board = game.change_values(next_board)
                        next_board = np.array(next_board,dtype = np.float32).reshape(1,4,4,16)
                        feed_dict = {a.single_dataset:next_board}
                        temp_scores = session.run(single_output,feed_dict=feed_dict)

                        max_qvalue = np.max(temp_scores)

                        # final labels
                        labels[con] = (labels[con] + a.gamma*max_qvalue)
                        break

                    if (np.array_equal(current_board,board)):
                        done = 'lose'

                # decrease the epsilon
                if((e > episodes // 2) or (a.epsilon > 0.1 and iterations % 2500 == 0)):
                    a.epsilon = a.epsilon / 1.005

                # change the matrix values and store them in memory
                prev_state = deepcopy(current_board)
                prev_state = game.change_values(prev_state)
                prev_state = np.array(prev_state,dtype=np.float32).reshape(1,4,4,16)
                replay_labels.append(labels)
                replay_memory.append(prev_state)

                # back-propagation
                if(len(replay_memory)>=max_replay):
                    back_loss = 0
                    batch_num = 0
                    z = list(zip(replay_memory,replay_labels))
                    np.random.shuffle(z)
                    np.random.shuffle(z)
                    replay_memory,replay_labels = zip(*z)

                    for i in range(0,len(replay_memory), BATCH_SIZE):
                        if(i + BATCH_SIZE>len(replay_memory)):
                            break

                        batch_data = deepcopy(replay_memory[i:i+BATCH_SIZE])
                        batch_labels = deepcopy(replay_labels[i:i+BATCH_SIZE])

                        batch_data = np.array(batch_data,dtype=np.float32).reshape(BATCH_SIZE,4,4,16)
                        batch_labels = np.array(batch_labels,dtype=np.float32).reshape(BATCH_SIZE,OUTPUT_UNIT)

                        feed_dict = {a.tf_batch_dataset: batch_data, a.tf_batch_labels: batch_labels}
                        _,l = session.run([optimizer,loss],feed_dict=feed_dict)
                        back_loss += l
                        batch_num +=1
                    back_loss /= batch_num
                    losses.append(back_loss)
                    print(loss)

                    #store the parameters in a dictionary
                    outcomes['conv1_layer1_weights'] = session.run(a.conv1_layer1_weights)
                    outcomes['conv1_layer2_weights'] = session.run(a.conv1_layer2_weights)
                    outcomes['conv2_layer1_weights'] = session.run(a.conv2_layer1_weights)
                    outcomes['conv2_layer2_weights'] = session.run(a.conv2_layer2_weights)
                    outcomes['fc_layer1_weights'] = session.run(a.fc_layer1_weights)
                    outcomes['fc_layer2_weights'] = session.run(a.fc_layer2_weights)
                    outcomes['fc_layer1_biases'] = session.run(a.fc_layer1_biases)
                    outcomes['fc_layer2_biases'] = session.run(a.fc_layer2_biases)

                    #make new memory
                    replay_memory = list()
                    replay_labels = list()

                iterations += 1

            if((e+1)%200 == 0):
                current_time = time.time()
                elapsed_time = current_time - start_time
                start_time = time.time()
                scores.append(total_score)
                if len(losses) > 0:
                    log = "Episode {}-{} finished in {} seconds. Total score: {}. Loss: {}.\n".format(e-199, e, elapsed_time, total_score, losses[-1])
                else:
                    log = "Episode {}-{} finished in {} seconds. Total score: {}.\n".format(e-199, e, elapsed_time, total_score)
                logs.append(log)
                print(log)

    return outcomes, scores, losses, logs


################################################################################################################

if __name__ == '__main__':
    '''
    train model
    '''

    outcomes, scores, losses, logs = train()
