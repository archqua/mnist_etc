import json
import os
import subprocess

import mlflow
import numpy as np
import onnx

import names


def main():
    onnx_model = onnx.load_model(names.full_model_weights(suffix=".onnx"))

    # https://github.com/mlflow/mlflow/issues/7819
    mlflow.set_tracking_uri("http://localhost:5000")
    batch_size = 32
    inp_example = np.empty((batch_size, 28, 28, 1), dtype=np.float32)
    outp_example = np.empty((batch_size, 10), dtype=np.float32)
    signature = mlflow.models.signature.infer_signature(inp_example, outp_example)
    with mlflow.start_run():
        model_info = mlflow.onnx.log_model(onnx_model, "onnx_model", signature=signature)

    model_uri = os.path.join("mlartifacts", "0", str(model_info.run_id))
    model_info_json = {
        "name": "mnist_ae_clsf",
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parameters": {
            # "uri": model_info.model_uri,
            "uri": os.path.join(model_uri, "artifacts", "onnx_model"),
        },
    }
    with open("model-settings.json", "w") as fp:
        json.dump(model_info_json, fp, indent=2)
        fp.write("\n")

    print("starting server")
    subprocess.run(["mlserver", "start", "."])


if __name__ == "__main__":
    main()
