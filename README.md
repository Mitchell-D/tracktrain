# tracktrain

`tracktrain` is an organized workflow for configuring, training, and
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
variable `XLA_FLAGS` to the cuda library's parent directory with:

```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
```

## installation

Install `tracktrain` as a package by cloning the repository and
adding the repository directory to `$PYTHONPATH` with:

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/tracktrain"
```

Next, register the package with pip by calling the following command:

```bash
pip install -e /path/to/tracktrain
```

The `-e` flag installs the package in editable mode so that custom
changes to the configuration can be imported immediately (for example
to add your own model making method to the ModelDir.model_builders
configuration). You can leave out the flag if you don't intend to make
any changes.

## documentation

<p align="center">
  <img width="768" src="https://github.com/Mitchell-D/tracktrain/blob/main/docs/dependency-graph.png" />
</p>

### model directories

__ModelDir.py__

`ModelDir` is a class abstracting a directory minimally containing
a config file sufficient to create a compilable Model object, and
providing methods for interfacing with the model's configuration,
training metrics, and trained weights.

__ModelSet.py__

### building and training models

__compile\_and\_train.py__

__model\_methods.py__

__VariationalEncoderDecoder.py__

### configuring arguments

__config.py__

__utils.py__
