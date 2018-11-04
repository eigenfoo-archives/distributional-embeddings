'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# TODO read data in... or something
# data = tf.placeholder(tf.float32, shape=[], name='data')

EMBED_DIM = 20
VOCAB_SIZE = 10000  # FIXME compute this from data shape
NUM_ITER = 100000

# Initialize embeddings
mu_center = tf.get_variable('mu_center', [VOCAB_SIZE, EMBED_DIM],
                            tf.float32, tf.random_normal_initializer)
mu_context = tf.get_variable('mu_context', [VOCAB_SIZE, EMBED_DIM],
                             tf.float32, tf.random_normal_initializer)
sigma_center = tf.get_variable('sigma_center', [VOCAB_SIZE, EMBED_DIM],
                               tf.float32, tf.ones_initializer)
sigma_context = tf.get_variable('sigma_center', [VOCAB_SIZE, EMBED_DIM],
                                tf.float32, tf.ones_initializer)

for it in range(NUM_ITERS):
    pass
