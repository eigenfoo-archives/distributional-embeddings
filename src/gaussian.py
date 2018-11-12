'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data_builder  # FIXME (Jonny) tf.data replaces data_builder


try:
    VOCAB_SIZE = int(sys.argv[1])
    EMBED_DIM = int(sys.argv[2])
    CONTEXT_SIZE = int(sys.argv[3])  # Same as number of negative samples!
    MARGIN = float(sys.argv[4])
    NUM_EPOCHS = int(sys.argv[5])
    CLIP_NORM = float(sys.argv[6])
    MINIMUM = float(sys.argv[7])
    MAXIMUM = float(sys.argv[8])
except IndexError:
    msg = ('\nUsage:\n\tpython gaussian.py VOCAB_SIZE EMBED_DIM '
           'CONTEXT_SIZE MARGIN NUM_EPOCHS CLIP_NORM MINIMUM MAXIMUM\n')
    print(msg)
    sys.exit()


def expected_likelihood(mu1, sigma1, mu2, sigma2):
    '''
    Evaluates expected likelihood between two Gaussians. All parameters are
    expected to be tf.Tensors with shape [D,].
    - Determinant of a diagonal matrix is the product of entries.
    - Quadratic form tf.transpose(x)*A*x with a diagonal A is
      tf.reduce_sum(tf.multiply(tf.diag(A), x**2))
    '''
    coeff = 1 / ((2*np.pi)**EMBED_DIM * tf.reduce_prod(sigma1 + sigma2))
    quad_form = tf.reduce_sum(tf.multiply(sigma1 + sigma2, (mu1 - mu2)**2))
    return coeff * tf.exp(-0.5 * quad_form)


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

# TODO (Jonny) check that tf.data code works
# TODO (Jonny) do not use data builder, but we need the word_dict.

center_id = tf.placeholder(tf.int32, [])
context_ids = tf.placeholder(tf.int32, [CONTEXT_SIZE])
negative_ids = tf.placeholder(tf.int32, [CONTEXT_SIZE])

# Initialize embeddings
mu = tf.get_variable('mu', [VOCAB_SIZE, EMBED_DIM],
                     tf.float32, tf.random_normal_initializer)
sigma = tf.get_variable('sigma', [VOCAB_SIZE, EMBED_DIM],
                        tf.float32, tf.ones_initializer)

# Look up embeddings
# FIXME (George) is there a better way of doing this?
center_mu = tf.nn.embedding_lookup(mu, center_id)
center_sigma = tf.nn.embedding_lookup(sigma, center_id)
context_mus = tf.nn.embedding_lookup(mu, context_ids)
context_sigmas = tf.nn.embedding_lookup(sigma, context_ids)
negative_mus = tf.nn.embedding_lookup(mu, negative_ids)
negative_sigmas = tf.nn.embedding_lookup(sigma, negative_ids)

# Source: https://stackoverflow.com/a/40543116/10514795
context_parameters = (context_mus, context_sigmas)
negative_parameters = (negative_mus, negative_sigmas)
positive_energy = tf.map_fn(
    lambda params: expected_likelihood(center_mu, center_sigma,
                                       params[0], params[1]),
    context_parameters)  # [CONTEXT_SIZE, ]
negative_energy = tf.map_fn(
    lambda params: expected_likelihood(center_mu, center_sigma,
                                       params[0], params[1]),
    negative_parameters)  # [CONTEXT_SIZE, ]

max_margins = tf.maximum(0.0, MARGIN - positive_energy + negative_energy)
loss = tf.reduce_mean(max_margins)

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# TODO (George) add tqdm somehow
while True:
    try:
        line = sess.run(next_line)
    except tf.errors.OutOfRangeError:
        print('End of dataset.')
        break

    # TODO (Jonny) need a function to get center, context and negative ids.
    # next_sample may take whatever arguments needed.
    center_id_, context_ids_, negative_ids_ = next_sample()

    # Train
    sess.run(train_step, feed_dict={center_id: center_id_,
                                    context_ids: context_ids_,
                                    negative_ids: negative_ids_})

    # Regularize means and covariance eigenvalues
    mu = tf.clip_by_norm(mu, CLIP_NORM)
    sigma = tf.maximum(MINIMUM, tf.minimum(MAXIMUM, sigma))

# Save embedding parameters as .npy files
mu_np = mu.eval(session=sess)
sigma_np = sigma.eval(session=sess)
np.save('mu.npy', mu_np)
np.save('sigma.npy', sigma_np)
