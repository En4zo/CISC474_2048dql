
DEPTH1 = 128
DEPTH2 = 128
BATCH_SIZE = 512
INPUT_UNIT = 16
HIDDEN_UNIT = 256
OUTPUT_UNIT = 4  # four actions could be taken in total
EXPAND_SIZE = 2 * 4 * DEPTH2 * 2 + 3 * 3 * DEPTH2 * 2 + 4 * 3 * DEPTH1 *2