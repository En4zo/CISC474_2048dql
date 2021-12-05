import numpy as np
import random
import math


"""
The file contains all the methods needed to create an appropriate environment for of the game
It has the functions to randomly create games, make legal moves and check the end state of each game
"""


# initialize a new game
def new_game():
    matrix = np.zeros([4, 4])
    return matrix


# add 2 or 4 in the matrix
def add(mat):
    empty_cells = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if(mat[i][j]==0):
                empty_cells.append((i,j))
    if(len(empty_cells)==0):
        return mat

    index_pair = empty_cells[random.randint(0,len(empty_cells)-1)]

    prob = random.random()
    if prob >= 0.9:
        mat[index_pair[0]][index_pair[1]]=4
    else:
        mat[index_pair[0]][index_pair[1]]=2
    return mat


# to check state of the game
def check_terminal(mat):
    for i in range(len(mat)-1): #intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1): #more elegant to use exceptions but most likely this will be their solution
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'no'

    for i in range(len(mat)): #check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'no'

    for k in range(len(mat)-1): #to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'no'

    for j in range(len(mat)-1): #check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'no'

    return 'lose'


# find the number of empty cells in the game matrix.
def find_empty(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i][j] == 0:
                count += 1
    return count


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])

    return np.transpose(mat)


def cover_up(mat):
    new = [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done


def merge(mat):
    done=False
    score = 0
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                score += mat[i][j]
                mat[i][j+1] = 0
                done = True
    return mat, done, score


# up move
def up(game):
    game = transpose(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done, temp[2]


# down move
def down(game):
    game=reverse(transpose(game))
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done, temp[2]


# left move
def left(game):
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    return game, done, temp[2]


# right move
def right(game):
    game = reverse(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done, temp[2]


controls = {0: up,
            1: left,
            2: right,
            3: down}


# convert the input game matrix into corresponding power of 2 matrix.
def change_values(matrix):
    power_mat = np.zeros(shape=(1, 4, 4, 16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if matrix[i][j] == 0:
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(matrix[i][j], 2))
                power_mat[0][i][j][power] = 1.0
    return power_mat


