from typing import Callable, Optional, Union

import tensorflow as tf

# fails smh
# import tqdm
from tqdm.autonotebook import tqdm


@tf.function
def bayessian_sisyphus_step(
    template: tf.Tensor,
    similarity_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    means: tf.Tensor,
    variances: tf.Tensor,
    n_samples: int = 256,
    n_best: int = 64,
    lr: float = 0.1,
) -> tuple[tf.Tensor, tf.Tensor]:
    assert n_best <= n_samples, "n_best can't be greater then n_samples"
    n_feats = means.shape[0]
    assert n_feats == variances.shape[0], "must supply same number of means and variances"

    samples = tf.random.normal(
        shape=(n_samples, n_feats),
        mean=means,
        stddev=tf.math.sqrt(variances),
    )
    similarities = similarity_fn(template[tf.newaxis, ...], samples)

    _, best_indices = tf.math.top_k(similarities, k=n_best)
    best_samples = tf.gather(samples, best_indices)
    loc_means = tf.math.reduce_mean(best_samples, axis=0)
    loc_vars = tf.math.reduce_variance(best_samples, axis=0)
    means_upd = (1.0 - lr) * means + lr * loc_means
    lm2 = loc_means * loc_means  # local
    gm2 = means * means  # global
    um2 = means_upd * means_upd  # updated
    vars_upd = lr * (loc_vars + lm2) + (1 - lr) * (variances + gm2) - um2
    return means_upd, vars_upd


@tf.function
def bayessian_reconstruction(
    template: tf.Tensor,
    similarity_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    means: tf.Tensor,
    variances: tf.Tensor,
    n_samples: int = 256,
    n_best: int = 64,
    lr: float = 0.1,
    n_iter: int = 1024,
    use_tqdm: bool = False,
    eps: float = 1e-04,
) -> tuple[tf.Tensor, tf.Tensor, int]:
    if use_tqdm:
        itr = tqdm(tf.range(n_iter))
    else:
        itr = tf.range(n_iter)

    iter_count = 0
    for i in itr:
        if _has_nan(means) or _has_nan(variances):
            # tf.function doesn't allow to return w/o else branch
            break
        if tf.reduce_max(variances) < eps or tf.reduce_min(variances) < 0:
            # tf.function doesn't allow to return w/o else branch
            break
        means, variances = bayessian_sisyphus_step(
            template, similarity_fn, means, variances, lr=0.1
        )
        iter_count += 1

    return means, variances, iter_count


@tf.function
def _has_nan(t):
    return tf.math.reduce_any(tf.math.is_nan(t))
