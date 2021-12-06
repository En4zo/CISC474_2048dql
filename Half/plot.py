
import matplotlib.pyplot as plt


'''
read a txt file to list of int or float
'''
def readfile(path, name, rounding):
    list1 = []
    with open(path + name + '.txt') as f:
        for line in f:
            if rounding:
                list1.append(round(float(line.strip('\n'))))
            else:
                list1.append(float(line.strip('\n')))
    return list1



'''
slice the episode
'''
def slice(list, length):
    if length == 1:
        return list
    list1 = []
    first_index = 0
    second_index = length-1
    while second_index < 20000:
        list1.append(sum(list[first_index:second_index])/length)
        first_index +=length
        second_index += length
    return list1
            

   

episode1 = []
num = 0

# Plot loss over training for 20000 episodes
losses = readfile(path='/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/weights/', name='losses', rounding=False)
for i in range(len(losses)):
    episode1.append(num)
    num = num + 20

episode = list(range(19999))
plt.plot(episode1,losses)

plt.xlabel("Episodes")
plt.ylabel("Losses")
plt.title("Losses over 20000 episodes")
plt.savefig("/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/plot/loss_over_episodes.png")
plt.show()


# Plot total score vs episodes
scores = slice(readfile(path='/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/weights/', name='scores', rounding=False),100)
print(scores)
episode2 = []
for i in range(len(scores)):
    episode2.append(num)
    num = num + 100
plt.plot(episode2,scores)
plt.xlabel("Episodes")
plt.ylabel("Scores")
plt.title("Scores over 20000 episodes")
plt.savefig("/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/plot/Scores_over_episodes.png")
plt.show()

# plot final game value over episode and the mean game value

play = slice(readfile(path='/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/played/', name='final_value_trained', rounding=False),1)

episode3 = []
for i in range(len(play)):
    episode3.append(num)
    num = num + 100
x= []
for i in range(len(play)):
    mean = sum(play)/len(play)
    print(mean)
    x.append(mean)
plt.plot(episode3,play,label = "play score")
plt.plot(episode3,x,label = "mean play score")
plt.xlabel("Episodes")
plt.ylabel("Play_value")
plt.title("Play_Value over 20000 episodes")
plt.legend()
plt.savefig("/Users/enzuomou/Desktop/cisc474_final_group/CISC474_2048dql/Half/plot/PlayValue_over_episodes.png")
plt.show()