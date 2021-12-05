import numpy as np
import random
import tensorflow as tf
from copy import deepcopy
import game
from agent import Agent
import math
import time

#loss
J = []

#scores
scores = []

#to store final parameters
final_parameters = {}

#number of episodes
M = 500
mem_capacity = 6000

a = Agent()

#for single example
single_output = a.model(a.single_dataset)

#for batch data
logits = a.model(a.tf_batch_dataset)

#loss
loss = tf.square(tf.subtract(a.tf_batch_labels,logits))
loss = tf.reduce_sum(loss,axis=1)
loss = tf.reduce_mean(loss)/2.0

#optimizer
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.compat.v1.train.exponential_decay(float(a.alpha), global_step, 1000, 0.90, staircase=True)
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

#to store states and lables of the game for training
#states of the game
replay_memory = list()

#labels of the states
replay_labels = list()


with tf.compat.v1.Session() as session:
    tf.compat.v1.global_variables_initializer().run()
    print("Initialized")

    epsilon = a.epsilon

    #for episode with max score
    maximum = -1
    episode = -1

    #total_iters
    total_iters = 1

    #number of back props
    back=0

    for ep in range(M):

        board = game.new_game()
        game.add(board)
        game.add(board)

        #whether episode finished or not
        finish = 'not over'

        #total_score of this episode
        total_score = 0

        #iters per episode
        local_iters = 1

        while(finish=='not over'):
            prev_board = deepcopy(board)

            #get the required move for this state
            state = deepcopy(board)
            state = game.change_values(state)
            state = np.array(state,dtype = np.float32).reshape(1,4,4,16)
            feed_dict = {a.single_dataset:state}
            control_scores = session.run(single_output,feed_dict=feed_dict)

            #find the move with max Q value
            control_buttons = np.flip(np.argsort(control_scores),axis=1)

            #copy the Q-values as labels
            labels = deepcopy(control_scores[0])

            #generate random number for epsilon greedy approach
            num = random.uniform(0,1)

            #store prev max
            prev_max = np.max(prev_board)

            #num is less epsilon generate random move
            if(num<epsilon):
                #find legal moves
                legal_moves = list()
                for i in range(4):
                    temp_board = deepcopy(prev_board)
                    temp_board,_,_ = game.controls[i](temp_board)
                    if(np.array_equal(temp_board,prev_board)):
                        continue
                    else:
                        legal_moves.append(i)
                if(len(legal_moves)==0):
                    finish = 'lose'
                    continue

                #generate random move.
                con = random.sample(legal_moves,1)[0]

                #apply the move
                temp_state = deepcopy(prev_board)
                temp_state,_,score = game.controls[con](temp_state)
                total_score += score
                finish = game.check_terminal(temp_state)

                #get number of merges
                empty1 = game.findemptyCell(prev_board)
                empty2 = game.findemptyCell(temp_state)

                if(finish=='not over'):
                    temp_state = game.add(temp_state)

                board = deepcopy(temp_state)

                #get next max after applying the move
                next_max = np.max(temp_state)

                #reward math.log(next_max,2)*0.1 if next_max is higher than prev max
                labels[con] = (math.log(next_max,2))*0.1

                if(next_max==prev_max):
                    labels[con] = 0

                #reward is also the number of merges
                labels[con] += (empty2-empty1)

                #get the next state max Q-value
                temp_state = game.change_values(temp_state)
                temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                feed_dict = {a.single_dataset:temp_state}
                temp_scores = session.run(single_output,feed_dict=feed_dict)

                max_qvalue = np.max(temp_scores)

                #final labels add gamma*max_qvalue
                labels[con] = (labels[con] + a.gamma*max_qvalue)

            #generate the the max predicted move
            else:
                for con in control_buttons[0]:
                    prev_state = deepcopy(prev_board)

                    #apply the LEGAl Move with max q_value
                    temp_state,_,score = game.controls[con](prev_state)

                    #if illegal move label = 0
                    if(np.array_equal(prev_board,temp_state)):
                        labels[con] = 0
                        continue

                    #get number of merges
                    empty1 = game.findemptyCell(prev_board)
                    empty2 = game.findemptyCell(temp_state)


                    temp_state = game.add(temp_state)
                    board = deepcopy(temp_state)
                    total_score += score

                    next_max = np.max(temp_state)

                    #reward
                    labels[con] = (math.log(next_max,2))*0.1
                    if(next_max==prev_max):
                        labels[con] = 0

                    labels[con] += (empty2-empty1)

                    #get next max qvalue
                    temp_state = game.change_values(temp_state)
                    temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                    feed_dict = {a.single_dataset:temp_state}
                    temp_scores = session.run(single_output,feed_dict=feed_dict)

                    max_qvalue = np.max(temp_scores)

                    #final labels
                    labels[con] = (labels[con] + a.gamma*max_qvalue)
                    break

                if(np.array_equal(prev_board,board)):
                    finish = 'lose'

            #decrease the epsilon value
            if((ep>10000) or (epsilon>0.1 and total_iters%2500==0)):
                epsilon = epsilon/1.005


            #change the matrix values and store them in memory
            prev_state = deepcopy(prev_board)
            prev_state = game.change_values(prev_state)
            prev_state = np.array(prev_state,dtype=np.float32).reshape(1,4,4,16)
            replay_labels.append(labels)
            replay_memory.append(prev_state)


            #back-propagation
            if(len(replay_memory)>=mem_capacity):
                back_loss = 0
                batch_num = 0
                z = list(zip(replay_memory,replay_labels))
                np.random.shuffle(z)
                np.random.shuffle(z)
                replay_memory,replay_labels = zip(*z)

                for i in range(0,len(replay_memory),BATCH_SIZE):
                    if(i + BATCH_SIZE>len(replay_memory)):
                        break

                    batch_data = deepcopy(replay_memory[i:i+BATCH_SIZE])
                    batch_labels = deepcopy(replay_labels[i:i+BATCH_SIZE])

                    batch_data = np.array(batch_data,dtype=np.float32).reshape(BATCH_SIZE,4,4,16)
                    batch_labels = np.array(batch_labels,dtype=np.float32).reshape(BATCH_SIZE,OUTPUT_UNIT)

                    feed_dict = {a.tf_batch_dataset: batch_data, a.tf_batch_labels: batch_labels}
                    _,l = session.run([optimizer,loss],feed_dict=feed_dict)
                    back_loss += l

                    print("Mini-Batch - {} Back-Prop : {}, Loss : {}".format(batch_num,back,l))
                    batch_num +=1
                back_loss /= batch_num
                J.append(back_loss)

                #store the parameters in a dictionary
                final_parameters['conv1_layer1_weights'] = session.run(a.conv1_layer1_weights)
                final_parameters['conv1_layer2_weights'] = session.run(a.conv1_layer2_weights)
                final_parameters['conv2_layer1_weights'] = session.run(a.conv2_layer1_weights)
                final_parameters['conv2_layer2_weights'] = session.run(a.conv2_layer2_weights)
                # final_parameters['conv1_layer1_biases'] = session.run(a.conv1_layer1_biases)
                # final_parameters['conv1_layer2_biases'] = session.run(a.conv1_layer2_biases)
                # final_parameters['conv2_layer1_biases'] = session.run(a.conv2_layer1_biases)
                # final_parameters['conv2_layer2_biases'] = session.run(a.conv2_layer2_biases)
                final_parameters['fc_layer1_weights'] = session.run(a.fc_layer1_weights)
                final_parameters['fc_layer2_weights'] = session.run(a.fc_layer2_weights)
                final_parameters['fc_layer1_biases'] = session.run(a.fc_layer1_biases)
                final_parameters['fc_layer2_biases'] = session.run(a.fc_layer2_biases)

                #number of back-props
                back+=1

                #make new memory
                replay_memory = list()
                replay_labels = list()


            if(local_iters%400==0):
                print("Episode : {}, Score : {}, Iters : {}, Finish : {}".format(ep,total_score,local_iters,finish))

            local_iters += 1
            total_iters += 1

        scores.append(total_score)
        print("Episode {} finished with score {}, result : {} board : {}, epsilon  : {}, learning rate : {} ".format(ep,total_score,finish,board,epsilon,session.run(learning_rate)))
        print()

        if((ep+1)%1000==0):
            print("Maximum Score : {} ,Episode : {}".format(maximum,episode))
            print("Loss : {}".format(J[len(J)-1]))
            print()

        if(maximum<total_score):
            maximum = total_score
            episode = ep
    print("Maximum Score : {} ,Episode : {}".format(maximum,episode))

################################################################################################################

path = r''
weights = ['conv1_layer1_weights','conv1_layer2_weights','conv2_layer1_weights','conv2_layer2_weights','fc_layer1_weights','fc_layer1_biases','fc_layer2_weights','fc_layer2_biases']
for w in weights:
    flatten = final_parameters[w].reshape(-1,1)
    file = open( w +'.csv','w')
    file.write('Sno,Weight\n')
    for i in range(flatten.shape[0]):
        file.write(str(i) +',' +str(flatten[i][0])+'\n')
    file.close()
    print(w + " written!")

