import os

artifacts = "artifacts"


def ae_training_examples(epoch, plus_one=True):
    return os.path.join(artifacts, f"ae_training_examples_{epoch + int(plus_one)}")


ae_weights = os.path.join(artifacts, "ae_weights.h5")
clsf_fc_weights = os.path.join(artifacts, "clsf_fc_weights.h5")
