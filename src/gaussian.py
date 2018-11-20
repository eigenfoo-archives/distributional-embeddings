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
                    help='Name of data file. Must be a TFRecord.')
parser.add_argument('vocab_size', type=int,
                    help=('Number of unique tokens in the vocabulary. Must '
                          'include the not-a-word token!'))
parser.add_argument('window_size', type=int,
                    help='Window size (i.e. "diameter" of window).')
parser.add_argument('embed_dim', type=int,
                    help='Dimensionality of the embedding space.')
parser.add_argument('batch_size', type=int, nargs='?', default=512,
                    help='Batch size.')
parser.add_argument('margin', type=float, nargs='?', default=1.0,
                    help='Margin in max-margin loss. Defaults to 1.')
parser.add_argument('num_epochs', type=int, nargs='?', default=100,
                    help='Number of epochs. Defaults to 100.')
parser.add_argument('C', type=float, nargs='?', default=20.0,
                    help='Maximum L2 norm of mu. Defaults to 20.')
parser.add_argument('m', type=float, nargs='?', default=1e-3,
                    help='Minimum covariance eigenvalue. Defaults to 1e-3.')
parser.add_argument('M', type=float, nargs='?', default=1e3,
                    help='Maximum covariance eigenvalue. Defaults to 1e3.')

args = parser.parse_args()

center_id = tf.placeholder(tf.int32, [None])
context_ids = tf.placeholder(tf.int32, [None, args.window_size])
negative_ids = tf.placeholder(tf.int32, [None, args.window_size])

# Data
features = {
    'center': tf.FixedLenFeature([], tf.int64),
    'context': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'negative': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
}
dataset = (tf.data.TFRecordDataset([args.data_file])
             .map(lambda x: tf.parse_single_example(x, features))
             .batch(args.batch_size))
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

# Initialize embeddings
mu = tf.get_variable('mu', [args.vocab_size, args.embed_dim],
                     tf.float32, tf.random_normal_initializer)
sigma = tf.get_variable('sigma', [args.vocab_size, args.embed_dim],
                        tf.float32, tf.ones_initializer)

# Look up embeddings
# [BATCH_SIZE, EMBED_DIM, 1]
center_mu = tf.expand_dims(tf.nn.embedding_lookup(mu, center_id), -1)
center_sigma = tf.expand_dims(tf.nn.embedding_lookup(sigma, center_id), -1)
# [BATCH_SIZE, EMBED_DIM, WINDOW_SIZE]
context_mus = tf.linalg.transpose(tf.nn.embedding_lookup(mu, context_ids))
context_sigmas = tf.linalg.transpose(tf.nn.embedding_lookup(sigma, context_ids))
negative_mus = tf.linalg.transpose(tf.nn.embedding_lookup(mu, negative_ids))
negative_sigmas = tf.linalg.transpose(tf.nn.embedding_lookup(sigma, negative_ids))

# Compute expected likelihood
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
with tf.control_dependencies([train_step]):
    clip_mu = tf.clip_by_norm(mu, args.C)
    bound_sigma = tf.maximum(args.m, tf.minimum(args.M, sigma))

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(args.num_epochs):
    data = sess.run(next_batch)
    sess.run([train_step, clip_mu, bound_sigma],
             feed_dict={center_id: data['center'],
                        context_ids: data['context'],
                        negative_ids: data['negative']})

# Save embedding parameters as .npy files
mu_np = mu.eval(session=sess)
sigma_np = sigma.eval(session=sess)
np.save('mu.npy', mu_np)
np.save('sigma.npy', sigma_np)
