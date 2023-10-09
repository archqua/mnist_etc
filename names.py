import os

artifacts = "artifacts"


def ae_training_examples(epoch, plus_one=True, prefix=""):
    return os.path.join(
        artifacts, prefix + f"ae_training_examples_{epoch + int(plus_one)}"
    )


def ae_weights(prefix=""):
    return os.path.join(artifacts, prefix + "ae_weights.h5")


def clsf_fc_weights(prefix=""):
    return os.path.join(artifacts, prefix + "clsf_fc_weights.h5")


def clsf_inference(prefix=""):
    return os.path.join(artifacts, prefix + "clsf_inference.csv")
