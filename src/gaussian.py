'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data_builder
from scipy.stats import multivariate_normal


try:
    VOCAB_SIZE = int(sys.argv[1])
    EMBED_DIM = int(sys.argv[2])
    CONTEXT_SIZE = int(sys.argv[3])
    NEGATIVE_SIZE = int(sys.argv[4])
    MARGIN = float(sys.argv[5])
    NUM_EPOCHS = int(sys.argv[6])
except IndexError:
    print('''\nUsage:\n\tpython gaussian.py VOCAB_SIZE EMBED_DIM
             CONTEXT_SIZE NEGATIVE_SIZE MARGIN NUM_EPOCHS\n''')
    sys.exit()


def expected_likelihood(mu1, sigma1, mu2, sigma2):
    '''
    Evaluates expected likelihood between two Gaussians.
    All parameters are expected to be 1d arrays.
    '''
    return multivariate_normal(mean=mu1-mu2,
                               cov=np.diag(sigma1+sigma2)).pdf(0.0)


def kl_divergence(mu1, sigma1, mu2, sigma2):
    '''
    Evaluates KL divergence of Gaussian 2 from Gaussian 1.
    All parameters are expected to be 1d arrays.
    '''
    return 0.5 * (np.sum(1/sigma1 * sigma2)
                  + (1/sigma1 * (mu1 - mu2)**2)
                  - EMBED_DIM
                  - np.sum(sigma2) + np.sum(sigma1))


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
context_mu = tf.nn.embedding_lookup(mu, context_ids)
context_sigma = tf.nn.embedding_lookup(sigma, context_ids)
negative_mu = tf.nn.embedding_lookup(mu, negative_ids)
negative_sigma = tf.nn.embedding_lookup(sigma, negative_ids)

# TODO Compute similarity here.

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
