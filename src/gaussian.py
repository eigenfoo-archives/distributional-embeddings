'''
Implementation of Word Representations via Gaussian Embedding.
    https://arxiv.org/abs/1412.6623
'''

import argparse
import numpy as np
import tensorflow as tf
from ast import literal_eval
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Gaussian embeddings.')

parser.add_argument('data_file', type=str,
                    help='Name of data file.')
parser.add_argument('vocab_size', type=int,
                    help='Number of unique tokens in the vocabulary.')
parser.add_argument('embed_dim', type=int,
                    help='Dimensionality of the embedding space.')
parser.add_argument('margin', type=float, nargs='?', default=1.0,
                    help='Margin in max-margin loss. Defaults to 1.')
parser.add_argument('num_epochs', type=int, nargs='?', default=1,
                    help='Number of epochs. Defaults to 1.')
parser.add_argument('C', type=float, nargs='?', default=20.0,
                    help='Maximum L2 norm of mu. Defaults to 20.')
parser.add_argument('m', type=float, nargs='?', default=1e-3,
                    help='Minimum covariance eigenvalue. Defaults to 1e-3.')
parser.add_argument('M', type=float, nargs='?', default=1e3,
                    help='Maximum covariance eigenvalue. Defaults to 1e3.')

args = parser.parse_args()

# FIXME (George) ensure that context_ids and negative_ids have the same shape
center_id = tf.placeholder(tf.int32, [])
context_ids = tf.placeholder(tf.int32, [None])
negative_ids = tf.placeholder(tf.int32, [None])

# Initialize embeddings
mu = tf.get_variable('mu', [args.vocab_size, args.embed_dim],
                     tf.float32, tf.random_normal_initializer)
sigma = tf.get_variable('sigma', [args.vocab_size, args.embed_dim],
                        tf.float32, tf.ones_initializer)

# Look up embeddings
# FIXME (George) is there a better way of doing this?
center_mu = tf.nn.embedding_lookup(mu, center_id)
center_sigma = tf.nn.embedding_lookup(sigma, center_id)
context_mus = tf.nn.embedding_lookup(mu, context_ids)
context_sigmas = tf.nn.embedding_lookup(sigma, context_ids)
negative_mus = tf.nn.embedding_lookup(mu, negative_ids)
negative_sigmas = tf.nn.embedding_lookup(sigma, negative_ids)

# Compute similarity (i.e. expected likelihood), max margin and loss
coeff_pos = 1 / ((2*np.pi)**args.embed_dim
                 * tf.reduce_prod(center_sigma + context_sigmas))
quadform_pos = \
    (center_mu - context_mus)**2 / (center_sigma + context_sigmas)**2
positive_energies = coeff_pos * tf.exp(-0.5 * quadform_pos)

coeff_neg = 1 / ((2*np.pi)**args.embed_dim
                 * tf.reduce_prod(center_sigma + negative_sigmas))
quadform_neg = \
    (center_mu - negative_mus)**2 / (center_sigma + negative_sigmas)**2
negative_energies = coeff_neg * tf.exp(-0.5 * quadform_neg)

max_margins = tf.maximum(0.0,
                         args.margin - positive_energies + negative_energies)
loss = tf.reduce_mean(max_margins)

# Minimize loss
train_step = tf.train.AdamOptimizer().minimize(loss)

# Regularize means and covariance eigenvalues
with tf.control_dependencies([train_step]) as control_deps:
    clip_mu = tf.clip_by_norm(mu, args.C)
    bound_sigma = tf.maximum(args.m, tf.minimum(args.M, sigma))

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(args.num_epochs):
    with open(args.data_file, 'r') as data_file:
        for line in tqdm(data_file.readlines()):
            # Evaluate string as python literal and convert to numpy array
            context_ids_, negative_ids_, center_id_ = \
                literal_eval(line.strip())
            context_ids_, negative_ids_, center_id_ = \
                map(np.array, [context_ids_, negative_ids_, center_id_])

            # Update
            sess.run([train_step, clip_mu, bound_sigma],
                     feed_dict={center_id: center_id_,
                                context_ids: context_ids_,
                                negative_ids: negative_ids_})

# Save embedding parameters as .npy files
mu_np = mu.eval(session=sess)
sigma_np = sigma.eval(session=sess)
np.save('mu.npy', mu_np)
np.save('sigma.npy', sigma_np)
