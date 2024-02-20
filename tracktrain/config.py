""" Configuration documentation and defaults for tracktrain modules """

""" --( Configuration for compile_and_train.py )-- """

compile_valid_args = (
        "model_name", "learning_rate", "metrics", "loss","weighted_metrics"
        )
compile_arg_descriptions = {
        "learning_rate":"Learning rate ceiling for Adam optimizer",
        "metrics":"List of Metric objects or metric names to track ",
        "weighted_metrics":"Metrics to scale by generated sample weights",
        "loss":"loss function to use for training"
        }
compile_arg_defaults = {
        "weighted_metrics":None,
        "loss":"mse"
        }

train_valid_args = (
        "model_name", "early_stop_metric", "early_stop_patience",
        "save_weights_only", "batch_size", "batch_buffer", "max_epochs",
        "val_frequency", "callbacks",
        )
train_arg_descriptions = {
        "model_name": \
                "Unique string name for this model, which must match the "
                "name of the model directory",
        "early_stop_metric":"\
                string metric evaluated during training to track "
                "learning stagnation and determine when to stop training",
        "early_stop_patience": \
                "int number of epochs before stopping",
        "save_weights_only": \
                "If True, ModelCheckpoint will only save model weights "
                "(as .weights.hdf5 files), instead of the full model "
                "metadata. This should be used if a custom class or loss "
                "function can't serialize",
        "batch_size": \
                "int minibatch size (samples per weight update)",
        "batch_buffer": \
                "int num of generator batches to preload in memory",
        "max_epochs": \
                "int maximum number of epochs to train",
        "val_frequency": \
                "int epochs between validations",
        }
train_arg_defaults = {
        "callbacks":["early_stop","model_checkpoint","csv_logger"],
        "early_stop_metric":"val_loss",
        "early_stop_patience":20,
        "save_weights_only":False,
        "batch_size":32,
        "batch_buffer":3,
        "max_epochs":512,
        "val_frequency":1,
        }

""" --( Configuration for VariationalEncoderDecoder.py )-- """

vae_valid_args = (
        "model_name", "num_inputs", "num_outputs", "num_latent",
        "enc_node_list", "dec_node_list", "dropout_rate", "batchnorm",
        "enc_dense_kwargs", "dec_dense_kwargs"
        )
vae_arg_descriptions = {
    "model_name":" String name of this model (for naming output files)",
    "num_inputs": "Number of inputs received by the model.",
    "num_outputs": "Number of outputs predicted by the model",
    "num_latent": "Dimensionality of the latent distribution.",
    "enc_node_list": "List of ints corresponding to the width of each" + \
            "encoder layer in number of nodes.",
    "dec_node_list":"List of ints corresponding to the width of each " + \
            "decoder layer in number of nodes.",
    "dropout_rate":"Ratio of random nodes disabled during training",
    "batchnorm": "If True, normalizes layer-wise activation amounts.",
    "enc_dense_kwargs": "dict of args to init encoder Dense layers.",
    "dec_dense_kwargs": "dict of args to init decoder Dense layers.",
    }
vae_arg_defaults = {
        "batchnorm":True,
        "dropout_rate":0.0,
        "enc_dense_kwargs":{},
        "dec_dense_kwargs":{},
        }

""" --( Configuration for model_methods.py )-- """

dense_kwargs_default = {
        "activation":"sigmoid",
        "use_bias":True,
        "bias_initializer":"zeros",
        "kernel_initializer":"glorot_uniform",
        "kernel_regularizer":None,
        "bias_regularizer":None,
        "activity_regularizer":None,
        "kernel_constraint":None,
        "bias_constraint":None,
        }


if __name__=="__main__":
    from itertools import chain
    """
    Unit test validating that all defaults and descriptions
    are paired with a valid argument.
    """
    ## supplemental dictionaries mapping arguments to descriptions or defaults.
    sup_dicts = (
            compile_arg_defaults,
            compile_arg_descriptions,
            train_arg_defaults,
            train_arg_descriptions,
            vae_arg_defaults,
            vae_arg_descriptions,
            )
    ## compine all tuples of valid arguments
    valid_args = tuple(chain(
        compile_valid_args,
        train_valid_args,
        vae_valid_args,
        ))
    sup_keys = tuple({k for d in sup_dicts for k in d.keys()})
    missing_keys = [k for k in sup_keys if k not in valid_args]
    if len(missing_keys):
        raise ValueError(
                f"Missing arguments have default or description: ",
                missing_keys
                )
    print("config valid 👍")
