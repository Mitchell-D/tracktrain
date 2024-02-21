# tracktrain

`tracktrain` is an abstract framework for configuring, training, and
evaluating Models within TensorFlow.

## capabilities

1. Basic methods for constructing generators and Model objects or
   their components in `model_methods.py` and
   `VariationalEncoderDecoder`,
2. Generalized methods for compiling and training models in
   `compile_and_train.py`, which rely on a dict with well-defined
   options defined in `config.py`.
3. Classes in `ModelDir.py` provide an interface for valid model
   configurations and training results stored in model directories
   abiding by standards defined and enforced by the
   `ModelDir.ModelDir` class. The `ModelSet` class abstracts a
   directory containing many model subdirectories, and enables the
   user to search and sort a set of `ModelDir.ModelDir` instances.

## dependencies

 - `python>=3.9`
 - `tensorflow>=12.4`
 - `numpy`
 - (python standard library)

If training on a GPU using cuda, you may need to set the environment
variable `XLA_FLAGS` to the package directory with:

```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda
```

## documentation

<p align="center">
  <img width="512" src="https://github.com/Mitchell-D/tracktrain/blob/main/tracktrain/docs/dependency-graph.png" />
</p>

### model directories

__ModelDir.py__

__ModelSet.py__

### building and training models

__compile\_and\_train.py__

__model\_methods.py__

__VariationalEncoderDecoder.py__

### configuring arguments

__config.py__

__utils.py__
