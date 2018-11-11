'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    VOCAB_SIZE = int(sys.argv[1])
    EMBED_DIM = int(sys.argv[2])
    NUM_EPOCHS = int(sys.argv[3])
except IndexError:
    print('\nUsage:\n\tpython gaussian.py VOCAB_SIZE EMBED_DIM NUM_EPOCHS\n')
    sys.exit()

'''
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
'''

# Initialize embeddings
mu = tf.get_variable('mu', [VOCAB_SIZE, EMBED_DIM],
                     tf.float32, tf.random_normal_initializer)
sigma = tf.get_variable('sigma', [VOCAB_SIZE, EMBED_DIM],
                        tf.float32, tf.ones_initializer)

while True:
    try:
        line = sess.run(next_line)
    except tf.errors.OutOfRangeError:
        print('End of dataset.')
        break

    # FIXME need to figure this out with Jonny. There are two parts to this:
    # getting the samples, and then converting them to unique integer ids.
    center_id, context_ids, negative_ids = next_sample()

    # Look up embeddings
    # FIXME we should perform only two embedding_lookups, and index.
    center_mu = tf.nn.embedding_lookup(mu, center_id)
    center_sigma = tf.nn.embedding_lookup(sigma, center_id)
    context_mu = tf.nn.embedding_lookup(mu, context_ids)
    context_sigma = tf.nn.embedding_lookup(sigma, context_ids)
    negative_mu = tf.nn.embedding_lookup(mu, negative_ids)
    negative_sigma = tf.nn.embedding_lookup(sigma, negative_ids)

    # TODO Train

    # TODO Regularize (e.g. eigenvalues of covariance matrix)
