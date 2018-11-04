# Ideas

- Use Dirichlet distribution instead of Gaussian.
  - This might involve deriving the updates mathematically
- Use a Gaussian with non-spherical or non-diagonal covariance matrices. Look
  into a low-rank covariance matrix and use the [Woodbury matrix
  identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).
- Try looking at multimodal distributions? How do we deal with polysemy?

# Evaluation

From the Gaussian Embedding paper:

- Specificity/uncertainty in embeddings.
- Entailment
- Directly learning asymmetric relationships
- Word similarity benchmarks

- Consider extrinsic evaluation? I.e. evaluate these embeddings on some task.

# Literature Review and Research

- [Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623)
- [Multimodal Word Distributions](https://arxiv.org/abs/1704.08424)
- [Expected Likelihood: Not a Good Metric for Gaussian Embeddings](https://ieeexplore.ieee.org/document/8356905)

