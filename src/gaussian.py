'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# TODO read data in... or something
# data = tf.placeholder(tf.float32, shape=[], name='data')

try:
    VOCAB_SIZE = int(sys.argv[1])
    EMBED_DIM = int(sys.argv[2])
    NUM_EPOCHS = int(sys.argv[3])
except IndexError:
    print('\nUsage:\n\tpython gaussian.py VOCAB_SIZE EMBED_DIM NUM_EPOCHS\n')
    sys.exit()

# Point Tensorflow to data file
filenames = ['../data/data.txt']
dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Filter out empty lines
dataset = dataset.flat_map(
    lambda filename: (tf.data.TextLineDataset(filename)
                        .filter(lambda line: tf.not_equal(line, '')))
    )

# Batch size of line 1 each. Repeat as many times as we have epochs.
dataset = dataset.batch(1).repeat(NUM_EPOCHS)
iterator = dataset.make_one_shot_iterator()
next_line = iterator.get_next()  # Usage: sess.run(next_line)

# Initialize embeddings
mu_center = tf.get_variable('mu_center', [VOCAB_SIZE, EMBED_DIM],
                            tf.float32, tf.random_normal_initializer)
sigma_center = tf.get_variable('sigma_center', [VOCAB_SIZE, EMBED_DIM],
                               tf.float32, tf.ones_initializer)
