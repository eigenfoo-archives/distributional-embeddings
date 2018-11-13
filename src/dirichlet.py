'''
Implementation of Dirichlet Embeddings.
'''

import argparse
import numpy as np
import tensorflow as tf
from ast import literal_eval
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Dirichlet embeddings.')

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
alpha = tf.get_variable('alpha', [args.vocab_size, args.embed_dim],
                        tf.float32, tf.constant_initializer(1/args.embed_dim))
gamma = tf.get_variable('gamma', [args.vocab_size],
                        tf.float32, tf.ones_initializer)
beta = tf.get_variable('beta', [args.vocab_size, args.embed_dim],
                       tf.float32, tf.random_normal_initializer)

# Look up embeddings
# FIXME (George) is there a better way of doing this?
center_alpha = tf.nn.embedding_lookup(alpha, center_id)
center_gamma = tf.nn.embedding_lookup(gamma, center_id)
center_beta = tf.nn.embedding_lookup(beta, center_id)
context_alpha = tf.nn.embedding_lookup(alpha, context_id)
context_gamma = tf.nn.embedding_lookup(gamma, context_id)
context_beta = tf.nn.embedding_lookup(beta, context_id)
negative_alpha = tf.nn.embedding_lookup(alpha, negative_id)
negative_gamma = tf.nn.embedding_lookup(gamma, negative_id)
negative_beta = tf.nn.embedding_lookup(beta, negative_id)

# TODO (George) implement some energy function here. Probably -KL?
max_margins = tf.maximum(0.0,
                         args.margin - positive_energies + negative_energies)
loss = tf.reduce_mean(max_margins)

# Minimize loss
train_step = tf.train.AdamOptimizer().minimize(loss)

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(arg.num_epochs):
    with open(args.data_file, 'r') as data_file:
        for line in tqdm(data_file.readlines()):
            # Evaluate string as python literal and convert to numpy array
            context_ids_, negative_ids_, center_id_ = \
                literal_eval(line.strip())
            context_ids_, negative_ids_, center_id_ = \
                map(np.array, [context_ids_, negative_ids_, center_id_])

            # Update
            sess.run(train_step, feed_dict={center_id: center_id_,
                                            context_ids: context_ids_,
                                            negative_ids: negative_ids_})

            # TODO (George) think about regularization. Also, project alpha to
            # stay on simplex!

# Save embedding parameters as .npy files
mu_np = mu.eval(session=sess)
sigma_np = sigma.eval(session=sess)
np.save('mu.npy', mu_np)
np.save('sigma.npy', sigma_np)
