'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data_builder


try:
    VOCAB_SIZE = int(sys.argv[1])
    EMBED_DIM = int(sys.argv[2])
    CONTEXT_SIZE = int(sys.argv[3])  # Same as number of negative samples!
    MARGIN = float(sys.argv[4])
    NUM_EPOCHS = int(sys.argv[5])
except IndexError:
    print('''\nUsage:\n\tpython gaussian.py VOCAB_SIZE EMBED_DIM
             CONTEXT_SIZE MARGIN NUM_EPOCHS\n''')
    sys.exit()


def expected_likelihood(mu1, sigma1, mu2, sigma2):
    '''
    Evaluates expected likelihood between two Gaussians.
    '''
    const = 1 / ((2*np.pi)**EMBED_DIM * tf.reduce_prod(sigma1+sigma2))
    exp = tf.exp(-0.5 * tf.reduce_sum(tf.multiply(sigma1+sigma2, (mu1-mu2)**2)))
    return const * exp


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

# FIXME need to make data stuff
# data = data_builder.Data()

center_id = tf.placeholder(tf.int32, [])
context_ids = tf.placeholder(tf.int32, [CONTEXT_SIZE])
negative_ids = tf.placeholder(tf.int32, [NEGATIVE_SIZE])

# Initialize embeddings
mu = tf.get_variable('mu', [VOCAB_SIZE, EMBED_DIM],
                     tf.float32, tf.random_normal_initializer)
sigma = tf.get_variable('sigma', [VOCAB_SIZE, EMBED_DIM],
                        tf.float32, tf.ones_initializer)

# Look up embeddings
center_mu = tf.nn.embedding_lookup(mu, center_id)
center_sigma = tf.nn.embedding_lookup(sigma, center_id)
context_mus = tf.nn.embedding_lookup(mu, context_ids)
context_sigmas = tf.nn.embedding_lookup(sigma, context_ids)
negative_mus = tf.nn.embedding_lookup(mu, negative_ids)
negative_sigmas = tf.nn.embedding_lookup(sigma, negative_ids)

# TODO Compute similarity here.
# foo = tf.map_fn(lambda )

loss = tf.maximum(0.0,
                  MARGIN
                  - expected_likelihood()
                  + expected_likelihood())

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while True:
    try:
        line = sess.run(next_line)
    except tf.errors.OutOfRangeError:
        print('End of dataset.')
        break

    # FIXME need to figure this out with Jonny. There are two parts to this:
    # getting the samples, and then converting them to unique integer ids.
    center_id, context_ids, negative_ids = next_sample()

    # TODO Train

    # TODO Regularize (e.g. eigenvalues of covariance matrix)
