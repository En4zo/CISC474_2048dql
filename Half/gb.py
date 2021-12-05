# This file includes the variables will be used in the program
DEPTH1 = 128  # the depth for the first convolutional layer
DEPTH2 = 128  # the depth for the second convolutional layer
BATCH_SIZE = 512  # the batch size
INPUT_UNIT = 16  # the layer of the first input
HIDDEN_UNIT = 256
OUTPUT_UNIT = 4  # four actions could be taken in total
# the number of input for the fully connected network
EXPAND_SIZE = 2 * 4 * DEPTH2 * 2 + 3 * 3 * DEPTH2 * 2 + 4 * 3 * DEPTH1 * 2
