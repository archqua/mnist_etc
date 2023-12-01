import os

artifacts = "artifacts"


def ae_(hid_dim=32, epochs=3, private=False):
    return f"ae_h{hid_dim}_e{epochs}" + ("_private" if private else "")


def ae_training_examples(
    epoch,
    hid_dim=32,
    epochs=3,
    private=False,
    plus_one=True,
    prefix="",
):
    return os.path.join(
        artifacts,
        prefix + ae_(hid_dim, epochs, private) + f"examples_{epoch + int(plus_one)}",
    )


def ae_weights(
    hid_dim=32,
    epochs=3,
    private=False,
    prefix="",
    suffix=".h5",
):
    return os.path.join(
        artifacts,
        prefix + ae_(hid_dim, epochs, private) + "_weights" + suffix,
    )


def clsf_(inp_dim=32, ae_epochs=3, ae_private=False, epochs=6, private=False):
    return (
        f"clsf_h{inp_dim}_ee{ae_epochs}"
        + ("_eprivate" if ae_private else "")
        + f"_e{epochs}"
        + ("_private" if private else "")
    )


def clsf_fc_weights(
    inp_dim=32,
    ae_epochs=3,
    ae_private=False,
    epochs=6,
    private=False,
    prefix="",
    suffix=".h5",
):
    return os.path.join(
        artifacts,
        prefix
        + clsf_(inp_dim, ae_epochs, ae_private, epochs, private)
        + "_fc_weights"
        + suffix,
    )


def clsf_inference(
    inp_dim=32,
    ae_epochs=3,
    ae_private=False,
    epochs=6,
    private=False,
    prefix="",
):
    return os.path.join(
        artifacts,
        prefix
        + clsf_(inp_dim, ae_epochs, ae_private, epochs, private)
        + "_inference.csv",
    )
