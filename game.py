# import numpy as np
# import random
# import math
#
#
# class Game:
#     def __init__(self):
#         self.matrix = self.new_game()
#
#     # initialize a new game
#     @staticmethod
#     def new_game():
#         return np.zeros([4, 4], dtype=int)
#
#     # count the number of empty boxes in the grid
#     def count_empty(self):
#         count = 0
#         for i in range(4):
#             for j in range(4):
#                 if self.matrix[i][j] == 0:
#                     count += 1
#         return count
#
#     # randomly add 2 or 4 into the game
#     def add(self):
#         count = self.count_empty()
#         if count == 0:
#             return  ##########################
#
#         while True:
#             w = random.randint(0, 15)
#
#             if self.matrix[w // 4][w % 4] == 0:
#                 break
#         if np.random.binomial(1, 0.1) == 1:
#             self.matrix[w // 4][w % 4] = 4
#         else:
#             self.matrix[w // 4][w % 4] = 2
#
#     # to check state of the game
#     def game_over(self):
#         count = self.count_empty()
#         if count != 0:
#             return  ###################
#         for i in range(3): #intentionally reduced to check the row on the right and below
#             for j in range(3):
#                 if self.matrix[i][j] == self.matrix[i+1][j] or self.matrix[i][j+1] == self.matrix[i][j]:
#                     return 'not over'
#                 if self.matrix[3][i] == self.matrix[3][i + 1]:
#                     return 'not over'
#                 if self.matrix[i][3] == self.matrix[i+1][3]:
#                     return 'not over'
#         return 'lose'
#
#
# def reverse(mat):
#     new=[]
#     for i in range(len(mat)):
#         new.append([])
#         for j in range(len(mat[0])):
#             new[i].append(mat[i][len(mat[0])-j-1])
#     return new
#
# def transpose(mat):
#     return np.transpose(mat)
#
# def cover_up(mat):
#     new = np.zeros([4, 4], dtype=int)
#     for i in range(4):
#         count = 0
#         for j in range(4):
#             if mat[i][j]!=0:
#                 new[i][count] = mat[i][j]
#                 count+=1
#     return new
#
# def merge(mat):
#     score = 0
#     visited = []
#     for i in range(4):
#          for j in range(3):
#              if (not(mat[i][j] in visited)) and (not(mat[i][j+1] in visited)) and mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
#                  mat[i][j]*=2
#                  score += mat[i][j]
#                  mat[i][j+1]=0
#                  visited.append(mat[i][j])
#                  visited.append(mat[i][j+1])
#     for i in range(3):
#          for j in range(4):
#              if (not(mat[i][j] in visited)) and (not(mat[i+1][j] in visited)) and mat[i+1][j]==mat[i][j] and mat[i][j]!=0:
#                  mat[i][j]*=2
#                  score += mat[i][j]
#                  mat[i+1][j]=0
#                  visited.append(mat[i][j])
#                  visited.append(mat[i+1][j])
#     return (mat,score)
#
# # up move
# def up(game):
#         game = transpose(game)
#         game, temp = update(game)
#         game = transpose(game)
#         return (game,temp)
#
# # down move
# def down(game):
#         game=reverse(transpose(game))
#         game, temp = update(game)
#         game=transpose(reverse(game))
#         return (game,temp)
#
# # left move
# def left(game):
#         game, temp = update(game)
#         return (game,temp)
#
# # right move
# def right(game):
#         game=reverse(game)
#         game, temp = update(game)
#         game=reverse(game)
#         return (game,temp)
#
# controls = {0:up,1:left,2:right,3:down}
#
# def update(game):
#     game = cover_up(game)
#     temp = merge(game)
#     game = temp[0]
#     game = cover_up(game)
#     return game, temp[1]
#
# def change_values(X):
#     power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
#     for i in range(4):
#         for j in range(4):
#             if(X[i][j]==0):
#                 power_mat[0][i][j][0] = 1.0
#             else:
#                 power = int(math.log(X[i][j],2))
#                 power_mat[0][i][j][power] = 1.0
#     return power_mat
#
#
import numpy as np
import random
import math

# initialize a new game
def new_game(n):
    matrix = np.zeros([n,n], dtype=int)
    return matrix

# to check state of the game
def isgameover(mat):
    #if 2048 in mat:
    #    return 'win'
    count = findemptyCell(mat)
    if count!=0:
        return mat
    for i in range(len(mat)-1): #intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1): #more elegant to use exceptions but most likely this will be their solution
            if mat[i][j]==mat[i+1][j] or mat[i][j+1]==mat[i][j]:
                return 'not over'
        if mat[len(mat) - 1][i] == mat[len(mat) - 1][i + 1]:
            return 'not over'
        if mat[i][len(mat)-1]==mat[i+1][len(mat)-1]:
            return 'not over'
    return 'lose'

# add 2 or 4 in the matrix
def randomfill(mat):
    count = findemptyCell(mat)
    if count==0:
        return mat
    empty = False
    while not empty:
        w = random.randint(0, 15)

        if mat[w // 4][w % 4] == 0:
            empty = True
    prob = random.random()
    if(prob>=0.9):
        mat[w // 4][w % 4]=4
    else:
        mat[w // 4][w % 4]=2
    return mat



def reverse(mat):
    new=[]
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

def transpose(mat):
    return np.transpose(mat)

def cover_up(mat):
    new = np.zeros([4, 4], dtype=int)
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count] = mat[i][j]
                count+=1
    return new

def merge(mat):
    score = 0
    visited = []
    for i in range(4):
        for j in range(3):
            if (not(mat[i][j] in visited)) and (not(mat[i][j+1] in visited)) and mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                mat[i][j]*=2
                score += mat[i][j]
                mat[i][j+1]=0
                visited.append(mat[i][j])
                visited.append(mat[i][j+1])
    for i in range(3):
        for j in range(4):
            if (not(mat[i][j] in visited)) and (not(mat[i+1][j] in visited)) and mat[i+1][j]==mat[i][j] and mat[i][j]!=0:
                mat[i][j]*=2
                score += mat[i][j]
                mat[i+1][j]=0
                visited.append(mat[i][j])
                visited.append(mat[i+1][j])
    return (mat,score)

# up move
def up(game):
    game = transpose(game)
    game, temp = update(game)
    game = transpose(game)
    return (game,temp)

# down move
def down(game):
    game=reverse(transpose(game))
    game, temp = update(game)
    game=transpose(reverse(game))
    return (game,temp)

# left move
def left(game):
    game, temp = update(game)
    return (game,temp)

# right move
def right(game):
    game=reverse(game)
    game, temp = update(game)
    game=reverse(game)
    return (game,temp)

controls = {0:up,1:left,2:right,3:down}

def update(game):
    game = cover_up(game)
    temp = merge(game)
    game = temp[0]
    game = cover_up(game)
    return game, temp[1]

def change_values(X):
    power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[0][i][j][power] = 1.0
    return power_mat

# find the number of empty cells in the game matrix.
def findemptyCell(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(mat[i][j]==0):
                count+=1
    return count