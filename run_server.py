import mlflow
import onnx

import names


def main():
    onnx_ae = onnx.load_model(names.ae_weights(suffix=".onnx"))
    onnx_fc = onnx.load_model(names.clsf_fc_weights(suffix=".onnx"))

    mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_tracking_uri("localhost:5000")
    with mlflow.start_run():
        ae_info = mlflow.onnx.log_model(onnx_ae, "onnx_ae")
        fc_info = mlflow.onnx.log_model(onnx_fc, "onnx_fc")

    ae = mlflow.pyfunc.load_model(ae_info.model_uri)
    fc = mlflow.pyfunc.load_model(fc_info.model_uri)

    _ = ae
    _ = fc


if __name__ == "__main__":
    main()
