# mnist_etc
This is for MLOps course.
Initially it was meant to explore private training --
I wanted to use it to build digit classifier that doesn't allow reconstructing
input image using model's outputs.
I didn't manage to do so and now it's just homework.

Configuration is stored via `dvc`.
Run `dvc pull conf.dvc` before doing anything.

Currently there are two train loops in `train_` directory:
`autoencoder.py` and `classifier.py` to train
a simple convolution-deconvolution autoencoder and
a semi-linear classifier, based on autoencoder's hidden representation.
These should be run as `[poetry run] python3 -m train_.<loop> [key=value list of hydra settings]`
from project's root directory or via `[poetry run] python3 train.py [key=value list of hydra settings]`.
Run classifier's training only when autoencoder's training has finished.

Resulting weights after two epochs are saved in `artifacts/ae_h<hid_dim>_e<epochs>[_private]_weights.<format>`
and `artifacts/clsf_h<inp_dim>_ee<AE epochs>[_eprivate]_e<epochs>[_private]_fc_weights.<format>`
with format being either `h5` or `onnx`.
During autoencoder's training example digit images -- original and reconstructed --
are saved under names `orig_{digit}.png` and `rec_{digit}.png` in
`artifacts/ae_h<hid_dim>_e<epochs>[_private]_examples_{epoch}`.

Inference is done via running `[poetry run] python3 infer.py [key=value list of hydra settings]` and
results are saved into `artifacts/clsf_h<inp_dim>_ee<AE epochs>[_eprivate]_e<epochs>[_private]_inference.csv`.

Reconstruction via
[Bayessian hill climbing](https://www.sciencedirect.com/science/article/abs/pii/S0031320309003380)
is examined in `noprotect_reconstruction.py`.
Run as `[poetry run] python3 noprotect_reconstruction.py`.


# System
- OS: Artix Linux x86\_64
- CPU: Intel i5-10310U (8) @ 4.400GHz
- vCPUs: 8
- Memory: 7542MiB


# Model repository
`model_repository` repository can be found in `triton` and has the following structure:
```
model_repository
└── onnx-mnist_etc
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```


# Performance
With following `config.pbtxt`
```
max_batch_size: 64
...
instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 500,
  preferred_batch_size: [ 8, 16, 32, 64 ]
  preserve_ordering: true
}

```
We get following performances when run
`perf_analyzer -m onnx-mnist_etc -u localhost:8500 --concurrency-range 3:6 --shape input:5,28,28`
```
Concurrency: 3, throughput: 3131.89 infer/sec, latency 957 usec
Concurrency: 4, throughput: 3954.11 infer/sec, latency 1010 usec
Concurrency: 5, throughput: 4826.92 infer/sec, latency 1035 usec
Concurrency: 6, throughput: 5604.94 infer/sec, latency 1069 usec
```

Then for `perf_analyzer -m onnx-mnist_etc -u localhost:8500 --concurrency-range 20:20 --shape input:5,28,28`
- for `max_queue_delay_microseconds: 250` we get `throughput: 7863.14 infer/sec, latency 2541 usec`
- for `max_queue_delay_microseconds: 500` we get `throughput: 11991.3 infer/sec, latency 1666 usec`
- for `max_queue_delay_microseconds: 2000` we get `throughput: 11383 infer/sec, latency 1755 usec`

After setting
```
instance_group [
  {
    count: 4
    kind: KIND_CPU
  }
]
```
for `perf_analyzer -m onnx-mnist_etc -u localhost:8500 --concurrency-range 20:20 --shape input:5,28,28`
- for `max_queue_delay_microseconds: 1` we get `throughput: 10194 infer/sec, latency 1960 usec`
- for `max_queue_delay_microseconds: 20` we get `throughput: 10120.6 infer/sec, latency 1975 usec`
- for `max_queue_delay_microseconds: 100` we get `throughput: 10446.2 infer/sec, latency 1913 usec`
- for `max_queue_delay_microseconds: 250` we get `throughput: 10493.5 infer/sec, latency 1904 usec`
- for `max_queue_delay_microseconds: 500` we get `throughput: 10416.4 infer/sec, latency 1919 usec`
- for `max_queue_delay_microseconds: 5000` we get `throughput: 9685.33 infer/sec, latency 2063 usec`
- for `max_queue_delay_microseconds: 10000` we get `throughput: 9685.33 infer/sec, latency 2063 usec`

After setting
```
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```
for `perf_analyzer -m onnx-mnist_etc -u localhost:8500 --concurrency-range 20:20 --shape input:5,28,28`
- for `max_queue_delay_microseconds: 10` we get `throughput: 19754.2 infer/sec, latency 1011 usec`
- for `max_queue_delay_microseconds: 250` we get `throughput: 19580.4 infer/sec, latency 1020 usec`
- for `max_queue_delay_microseconds: 500` we get `throughput: 19758.5 infer/sec, latency 1011 usec`
- for `max_queue_delay_microseconds: 2000` we get `throughput: 19727 infer/sec, latency 1012 usec`

## Conclusion
These results are rather weird: one cpu is optimal and
`max_queue_delay_microseconds` matters only for 2 cpus.

I didn't want to increase max batch size, because allowing heavy operations seems not quite OK.
Heavier loads can be handled using more model instances.


# Dependencies

## Build
- `poetry`

## Run
- `python` version `>=3.9,<3.12`
- `tensorflow` version `<2.14.0`
- `tqdm` version `^4.66.1`
- `pillow` version `^10.0.1`
- `tensorflow-privacy` version `<0.8.11`
- `dvc` version `^3.30.1`
- `dvc-gdrive` version `^2.20.0`
- `hydra-core` version `^1.3.2`
- `mlflow` version `^2.8.1`
- `onnx` version `<1.15.0`
- `tf2onnx` version `^1.15.1`


# TODO
- automatic docgen with sphinx and github actions
- `tensorflow.privacy` to break hill climbing reconstruction
    (currently it doesn't break reconstruction)
- remove tesnorflow import when running (mlflow) inference server
