'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import sys
import numpy as np
import tensorflow as tf
from ast import literal_eval
from tqdm import tqdm


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

# Expected likelihood
coeff_pos = \
    1 / ((2*np.pi)**EMBED_DIM * tf.reduce_prod(center_sigma + context_sigmas))
quadform_pos = \
    (center_mu - context_mus)**2 / (center_sigma + context_sigmas)**2
positive_energies = coeff_pos * tf.exp(-0.5 * quadform_pos)

coeff_neg = \
    1 / ((2*np.pi)**EMBED_DIM * tf.reduce_prod(center_sigma + negative_sigmas))
quadform_neg = \
    (center_mu - negative_mus)**2 / (center_sigma + negative_sigmas)**2
negative_energies = coeff_neg * tf.exp(-0.5 * quadform_neg)

max_margins = tf.maximum(0.0, MARGIN - positive_energies + negative_energies)
loss = tf.reduce_mean(max_margins)

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with open('sample_data.txt', 'r') as data_file:
    for line in tqdm(data_file.readlines()):
        # Evaluate string as python literal and convert to numpy array
        context_ids, negative_ids, center_id = literal_eval(line.strip())
        context_ids, negative_ids, center_id = \
            map(np.array, [context_ids, negative_ids, center_id])

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
