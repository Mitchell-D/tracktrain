"""
Two methods for compiling and training any valid tensorflow Model and Dataset
and saving the results in a directory following the conventions set by
tracktrain.ModelDir

 --( meta-information )--

"model_name": unique string name for this model. This value determines the
    name of the model directory, and the string prepended to its file names.
"data_source": string path to the data source. This value isn't used in the
    training process, but documenting it is a best practice
"notes": Optional (but recommended) string describing this model iteration

 --( exclusive to compile_and_build_dir )--

"model_name":unique string name for this model. This value determines the
    name of the model directory, and the string prepended to file names.
"learning_rate": float learning rate of the model
"loss": String representing the loss function to use per keras labels
"metrics": List of strings representing metrics to record per keras labels
"weighted_metrics": List of metric labels to  be weighted by masking level

--( exclusive to train )-

"early_stop_metric": string metric evaluated for stagnation
"early_stop_patience": int number of epochs before stopping
"save_weights_only": If True, ModelCheckpoint will only save model weights
    (as .weights.hdf5 files), instead of the full model metadata. This should
    be used if a custom class or loss function can't serialize
"batch_size": int minibatch size (samples per weight update)
"batch_buffer": int num of batches to preload in memory from the generator
"max_epochs": int maximum number of epochs to train
"val_frequency": int epochs between validations

The result is stored in the unique directory created by compile_and_build_dir;
It will contain the best-performing models as hdf5s at checkpoints,
and the following files:

 - "{model-name}_config.json":  serialized version of the user-defined config
 - "{model-name}_summary.txt":  serialized version of the user-defined config
 - "{model-name}_prog.csv":     keras CSVLogger output of learning curves.
"""
from pathlib import Path

from random import random
import pickle as pkl
import json
import numpy as np
import os
import sys

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model
from typing import Iterator

from tracktrain.utils import validate_keys
from tracktrain.config import compile_valid_args, compile_arg_descriptions
from tracktrain.config import train_valid_args, train_arg_descriptions
from tracktrain.config import compile_arg_defaults, train_arg_defaults


## Define the arguments that must appear in the configuration
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], True)
'''
def compile_from_config(model, compile_config:dict,
        custom_losses:dict={}, custom_metrics:dict={},):
    """
    Given an initialized Model object, Verify that the keys required to specify
    a loss function and any metrics are present and map to valid options, then
    compile the model.

    --( required configuration arguments )--

    "learning_rate":
        float learning rate of the model
    "loss":
        String representing the loss function to use per keras labels
    "metrics":
        List of strings representing metrics to record per keras labels

    :@param model:
    :@param compile_config:
    :@param custom_losses: Dict mapping unique strings to valid custom loss
        functions with argument signature (labels, predictions)
    :@param custom_metrics: Dict mapping unique keys to custom metric functions
    """
    compile_config = {**compile_arg_defaults, **compile_config}
    validate_keys(
            mandatory_keys=compile_valid_args,
            received_keys=list(compile_config.keys()),
            source_name="compile",
            descriptions=compile_arg_descriptions,
            )

    ## Deserialize string args  for built-in or custom metric and loss funcs.
    metrics = []
    for m in compile_config.get("metrics"):
        tmp_metric = tf.keras.metrics.deserialize(
                m, custom_objects=custom_metrics,)
        if type(tmp_metric) == str:
            raise ValueError(f"{m} is not a valid metric argument")
        metrics.append(tmp_metric)

    loss_fn = tf.keras.losses.deserialize(
            compile_config.get("loss"),
            custom_objects=custom_losses,
            )
    if type(loss_fn) is str:
        raise ValueError(f"{loss_fn} is not a valid loss function argument")

    opt = tf.keras.optimizers.deserialize({
        "class_name":compile_config.get("optimizer"),
        "config":{ "learning_rate":compile_config.get("learning_rate") },
        })

    ## Compile the model according to the provided args. assume Adam for
    ## optimizer, though may want to look into sequence optimization later
    model.compile(
            optimizer=opt,
            metrics=metrics,
            loss=loss_fn,
            ## weighted metrics no longer supported; use the third argument
            ## in a custom loss function to apply weights.
            #weighted_metrics=compile_config.get("weighted_metrics"),
            )
    return model

def compile_and_build_dir(
        model, model_parent_dir:Path, config:dict,
        custom_losses={}, custom_metrics={}, print_summary=True):
    """
    Run the model build pipeline, which compiles a Model object that has
    already been initialized, and creates a ModelDir - style directory with
    a {model_name}_summary.txt and {model_name}_config.json

    (1) Verify that all required config values were provided
    (2) Compile the model with Adam and the requested metrics
    (3) Create a new model directory with the model's configured "model_name"
    (4) Write a _summary.txt and _config.json file to the model directory

    --( mandatory config elements for this method )--

    {
    "model_name":str
    "compile":{
        "learning_rate":float,
        "loss":str,
        "metrics":list,
        }
    }

    :@param model: Initialized Model object that hasn't yet been compiled.
    :@param config: Configuration dictionary defining the above terms.
    :@param model_parent_dir: Parent directory where this model's dir will be
    :@param custom_losses: Dictionary of valid custom loss functions
    :@param custom_metrics: Dictionary of valid custom metric functions

    :@return: 2-tuple (model, model_dir_path)
    """
    compile_config = config.get("compile")
    if compile_config is None:
        raise ValueError(f"Provided configuration dict must have a " + \
                "sub-dict 'compile' with the documented default elements")
    model = compile_from_config(model, compile_config,
            custom_losses=custom_losses, custom_metrics=custom_metrics)
    ## Once we know that the model can compile, create and add some
    ## information to the dedicated model directory
    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError(f"Must define model_name in config dict")
    model_dir_path = model_parent_dir.joinpath(model_name)
    assert not model_dir_path.exists(), model_dir_path
    model_dir_path.mkdir()
    model_json_path = model_dir_path.joinpath(
            f"{config.get('model_name')}_config.json")
    model_json_path.open("w").write(json.dumps(config,indent=4))

    ## Write a summary of the model to a file
    if print_summary:
        ## Write a model summary to stdout and to a file
        model.summary(expand_nested=True)
    summary_path = model_dir_path.joinpath(
            config.get("model_name")+"_summary.txt")
    with summary_path.open("w") as f:
        model.summary(expand_nested=True,
                      print_fn=lambda x: f.write(x + '\n'))

    ## Return the compiled model
    return model,model_dir_path

def train(model_dir_path, train_config:dict, compiled_model:Model,
          gen_training:Iterator, gen_validation:Iterator,
          custom_lr_schedulers:dict={}):
    """
    Execute the training pipeline according to the provided configuration
    to fit the compiled_model using a TensorFlow Dataset instance returned by
    tf.data.Dataset.from_generator

    (1) Ensure all mandatory arguments are provided or have defaults
    (2) Initialize the callbacks
    (3) Fit the model using the data generators and configuration values
    (4) Optimize for the loss function, and save intermediate models that
        minimize the metric specified by "early_stop_metric"
    (5) Stop when early_stop_metric stagnates for early_stop_patience epochs

     --( mandatory config entries )--

    "early_stop_metric": string metric evaluated for stagnation
    "early_stop_patience": int number of epochs before stopping
    "save_weights_only": If True, ModelCheckpoint will only save model weights
        (as .weights.hdf5 files), instead of the full model metadata. This
        should be used if a custom class or loss function can't serialize.
    "batch_size": int minibatch size (samples per weight update)
    "batch_buffer": int num of batches to preload in memory from the generator
    "max_epochs": int maximum number of epochs to train
    "val_frequency": int epochs between validations

     --( default config entries )--

    "callbacks":["early_stop","model_checkpoint","csv_logger"],
    "early_stop_metric":"val_loss",
    "early_stop_patience":20,
    "save_weights_only":False,
    "batch_size":32,
    "batch_buffer":3,
    "max_epochs":512,
    "val_frequency":1,


    :@param model_dir_path: Existing dir with a _summary.txt &  _config.json
    :@param train_config: dict minimally containing the no-default mandatory
        training arguments listed above.
    :@param gen_training: tf.data.Dataset (only tested using from_generator)
    :@pram gen_validation: tf.data.Dataset (only tested using from_generator)
    :@param custom_lr_schedulers: Dict mapping unique keys to learning rate
        scheduler functions, which map epoch indeces and the current learning
        rate to a new learning rate.
    """
    assert model_dir_path.exists()
    #assert model_dir_path.name == train_config.get("model_name")
    model_name = model_dir_path.name
    ## train_config has order precedence in the dict re-composition
    train_config = {**train_arg_defaults, **train_config}
    ## Make sure the mandatory keys (or defaults) are in the dictionary
    validate_keys(
            mandatory_keys=train_valid_args,
            received_keys=list(train_config.keys()),
            source_name="train",
            descriptions=train_arg_descriptions,
            )
    ## Choose the model save file path based on whether only weights are stored
    halt_metric = train_config.get("early_stop_metric")
    if train_config.get("save_weights_only"):
        out_path = model_dir_path.joinpath(
                model_name +
                "_{epoch:03}_{loss:.3f}.weights.h5"
                )
    else:
        out_path = model_dir_path.joinpath(
                model_name+"_{epoch:03}_{loss:.03f}.h5"
                )

    ## Mandatory callbacks for ModelDir system
    callbacks = {
            "early_stop":tf.keras.callbacks.EarlyStopping(
                monitor=halt_metric,
                patience=train_config.get("early_stop_patience")
                ),
            "model_checkpoint":tf.keras.callbacks.ModelCheckpoint(
                monitor=halt_metric,
                save_best_only=True,
                filepath=out_path.as_posix(),
                save_weights_only=train_config.get("save_weights_only"),
                ),
            "csv_logger":tf.keras.callbacks.CSVLogger(
                model_dir_path.joinpath(
                    f"{model_name}_prog.csv"),
                ),
            }

    if not train_config.get("lr_scheduler") is None:
        lrs = custom_lr_schedulers[train_config.get("lr_scheduler")]
        callbacks.update({
            "lr_scheduler":tf.keras.callbacks.LearningRateScheduler(lrs)
            })

    train_data = gen_training.batch(train_config.get("batch_size"))
    train_data = train_data.prefetch(train_config.get("batch_buffer"))
    val_data = gen_validation.batch(train_config.get("batch_size"))
    val_data = val_data.prefetch(train_config.get("batch_buffer"))

    if train_config.get("repeat_data"):
        train_data = train_data.repeat()
        val_data = val_data.repeat()

    ## Train the model on the generated tensors
    hist = compiled_model.fit(
        train_data,
        epochs=train_config.get("max_epochs"),
        ## Number of batches to draw per epoch. Use full dataset by default
        #steps_per_epoch=train_config.get("train_steps_per_epoch"),
        validation_data=val_data,
        ## batches of validation data to draw per epoch
        #validation_steps=train_config.get("val_steps_per_epoch"),
        ## Number of epochs to wait between validation runs.
        validation_freq=train_config.get("val_frequency"),
        callbacks=callbacks,
        steps_per_epoch=train_config.get("steps_per_epoch"),
        validation_steps=train_config.get("validation_steps"),
        verbose=2,
        )

    ## Save the most performant model from the last checkpoint
    ## (!!!) This relies on the checkpoint file name formatting string, (!!!)
    ## (!!!) and on that there are no non-model .hdf5 files in the dir. (!!!)
    best_model = list(sorted(
        [q for q in model_dir_path.iterdir() if ".h5" in q.suffix],
        key=lambda p:int(p.stem.split("_")[1])
        )).pop(-1)
    ## Call the suffix so Path is {model_name}_final( .hdf5 | .weights.hdf5 )
    ## conditional on if the full Model object or just the weights are stored.
    best_model.rename(model_dir_path.joinpath(
        model_name+"_final"+"".join(
            [s for s in best_model.suffixes if s in (".weights",".hdf5",".h5")]
            )))
    return best_model

if __name__=="__main__":
    """ Example config for vae or ff """
    config = {
            ## Meta-info
            "model_name":"",
            "num_inputs":16,
            "num_outputs":4,
            "data_source":"data.hdf5",

            ## Exclusive to feedforward
            "node_list":[64,32,32,32,16,16],
            "dense_kwargs":{"activation":"sigmoid"},

            ## Exclusive to variational encoder-decoder
            "num_latent":4,
            "enc_node_list":[128,128,64,32],
            "dec_node_list":[16,16],
            "dropout_rate":0.1,
            "batchnorm":True,
            "enc_dense_kwargs":{"activation":"relu"},
            "dec_dense_kwargs":{"activation":"relu"},

            ## Common to models
            "batchnorm":True,
            "dropout_rate":0.1,

            ## Exclusive to compile_and_build_dir
            "learning_rate":1e-5,
            "loss":"mse",
            "metrics":["mse", "mae"],
            "weighted_metrics":["mse", "mae"],

            ## Exclusive to train
            "early_stop_metric":"val_mse", ## metric evaluated for stagnation
            "early_stop_patience":30, ## number of epochs before stopping
            "save_weights_only":True,
            "batch_size":32,
            "batch_buffer":4,
            "max_epochs":128, ## maximum number of epochs to train
            "val_frequency":1, ## epochs between validation

            ## Exclusive to generator init
            "train_val_ratio":.8,
            "mask_pct":0.0,
            "mask_pct_stdev":0.0,
            "mask_val":9999.,
            "mask_feat_probs":None,

            "notes":"",
            }
