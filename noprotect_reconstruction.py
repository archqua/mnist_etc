import argparse
import os

# fails smh
# import tqdm
from tqdm.autonotebook import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf  # noqa
from tensorflow.keras.losses import cosine_similarity  # noqa

import names  # noqa
from hill_climbing import bayessian_reconstruction  # noqa
from models import Autoencoder, Linear  # noqa


def main(use_tf_privacy=False):
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    @tf.function
    def _preproc(images, labels):
        return tf.cast(images, tf.float32)[..., tf.newaxis] / 255.0, labels

    batch_size = 64
    train_data = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(X_train.shape[0])
        .map(_preproc, num_parallel_calls=batch_size)
        .batch(batch_size)
    )
    val_data = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .shuffle(X_val.shape[0])
        .map(_preproc, num_parallel_calls=batch_size)
        .batch(batch_size)
    )

    prefix = "tf_private_" if use_tf_privacy else ""
    if not os.path.exists(names.artifacts):
        raise FileNotFoundError(
            f"directory `{names.artifacts}` must exist and contain"
            + f"`{os.path.basename(names.ae_weights())}` after running train/autoencoder"
        )
    if not os.path.exists(names.ae_weights(prefix="")):
        raise FileNotFoundError(f"file `{names.ae_weights()}` not found")
    if not os.path.exists(names.clsf_fc_weights(prefix=prefix)):
        raise FileNotFoundError(
            f"file `{names.clsf_fc_weights(prefix=prefix)}` not found"
        )
    hid_dim = 32
    ae = Autoencoder()
    ae.build(input_shape=(batch_size, 28, 28, 1))
    ae.load_weights(names.ae_weights(prefix=""))
    clsf = Linear(activation=None)
    clsf.build(input_shape=(batch_size, hid_dim))
    clsf.load_weights(names.clsf_fc_weights(prefix=prefix))

    @tf.function
    def _mse(trg, inf, axis=-1):
        return tf.squeeze(tf.math.reduce_mean((trg - inf) ** 2, axis=axis))

    def _similarityFn(penalty, transf=clsf, logarithmic=True):
        def wrapped(template, samples):
            similarity = -penalty(template[tf.newaxis, ...], transf(samples))
            if logarithmic:
                return -tf.math.log(-similarity)
            else:
                return similarity

        return wrapped

    # _mse_sim = _similarityFn(_mse)
    @tf.function
    @_similarityFn
    def _mse_sim(template, samples):
        return _mse(template, samples)

    print("calculating means")
    X_hid_mean = tf.zeros((hid_dim,))
    for X_batch, _ in tqdm(train_data):
        batch_hid = ae.encode(X_batch)
        batch_sum = tf.math.reduce_sum(batch_hid, axis=0)
        X_hid_mean += batch_sum
    X_hid_mean /= X_train.shape[0]
    print("calculating variances")
    X_hid_var = tf.zeros((hid_dim,))
    for X_batch, _ in tqdm(train_data):
        batch_hid = ae.encode(X_batch)
        X_hid_var += tf.math.reduce_sum(
            (batch_hid - X_hid_mean[tf.newaxis, ...]) ** 2, axis=0
        )
    X_hid_var /= X_train.shape[0]
    # X_hid_std = tf.math.sqrt(X_hid_var)

    print("reconstructing samples")
    for images, labels in val_data.take(1):
        img_hid = ae.encode(images)
        img_scores = clsf(img_hid)
        hid_rec = []
        for i in tqdm(range(img_hid.shape[0])):
            scores_template = img_scores[i, ...]
            rec, variance, iter_count = bayessian_reconstruction(
                template=scores_template,
                similarity_fn=_mse_sim,
                means=X_hid_mean,
                variances=X_hid_var,
                n_samples=256,
                n_best=64,
                lr=0.1,
                n_iter=1024,
                use_tqdm=False,
                eps=1e-04,
            )
            hid_rec.append(rec)
        hid_rec = tf.stack(hid_rec)

        fo_cos_similarities = -cosine_similarity(img_hid, hid_rec)
        visual_rec = ae.decode(hid_rec)
        direct_cos_similarities = -cosine_similarity(images, visual_rec)
        so_hid_rec = ae.encode(visual_rec)
        so_cos_similarities = -cosine_similarity(img_hid, so_hid_rec)
        print("cosine similarities:")
        print(f"first order perceptual: \t{tf.math.reduce_mean(fo_cos_similarities):.3f}")
        print(
            f"reconstruction direct:  \t{tf.math.reduce_mean(direct_cos_similarities):.3f}"
        )
        print(f"second order perceptual:\t{tf.math.reduce_mean(so_cos_similarities):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayessian hill-climbing digit image reconstruction"
    )
    parser.add_argument(
        "-p",
        "--private",
        action="store_true",
        dest="use_tf_privacy",
        default=False,
    )
    args = parser.parse_args()
    main(use_tf_privacy=args.use_tf_privacy)
