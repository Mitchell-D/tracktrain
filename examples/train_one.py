import numpy as np
import tensorflow as tf
from pathlib import Path

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

""" Configuration for data source """

data_dir = Path(".")
##Directory with sub-directories for each model.
model_parent_dir = data_dir.joinpath("models")
#asos_path = data_dir.joinpath("AL_ASOS_July_2023.csv")
asos_path = data_dir.joinpath("UT_ASOS_Mar_2023.csv")

input_feats = ["tmpc","dwpc","relh","sknt", "mslp","p01m","gust","feel"]
#input_feats = ["tmpc","relh","sknt","mslp"]
output_feats = ["romps_LCL_m","lcl_estimate"]

config = {
        ## Meta-info
        "model_name":"test-0",
        "num_inputs":len(input_feats),
        "num_outputs":len(output_feats),
        "data_source":asos_path.as_posix(),
        "input_feats":input_feats,
        "output_feats":output_feats,
        "model_type":"ved",
        "rand_seed":20240128,

        ## Exclusive to feedforward
        "node_list":[16,16,8,8,8,8,4],
        "dense_kwargs":{"activation":"relu"},

        ## Exclusive to variational encoder-decoder
        "num_latent":8,
        "enc_node_list":[64,128,64,16],
        "dec_node_list":[16,64,128,64],
        "enc_dense_kwargs":{"activation":"relu"},
        "dec_dense_kwargs":{"activation":"relu"},

        ## Common to models
        "batchnorm":True,
        "dropout_rate":0.05,

        ## Exclusive to compile_and_build_dir
        "learning_rate":1e-5,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":600, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":64,
        "batch_buffer":4,
        "max_epochs":2048, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        "train_val_ratio":.85,
        "mask_pct":0.0,
        "mask_pct_stdev":0.0,
        "mask_val":9999,
        "mask_feat_probs":None,

        "notes":"basic feedforward for testing",
        }

"""
--( Exclusive to generator init for testing )--

"train_val_ratio": Ratio of training to validation samples during training
"mask_pct": Float mean percentage of inputs to mask during training
"mask_pct_stdev": Stdev of number of inputs masked during training
"mask_val": Number substituted for masked features
"mask_feat_probs": List of relative probabilities of each feat being masked
"""

## Extract and preprocess the data
from preprocess import preprocess
data_dict = preprocess(
        asos_csv_path=asos_path,
        input_feats=input_feats,
        output_feats=output_feats,
        normalize=True,
        )
## Initialize the masking data generators
gen_train,gen_val = mm.array_to_noisy_tv_gen(
        X=data_dict["X"].astype(np.float64),
        Y=data_dict["Y"].astype(np.float64),
        tv_ratio=config.get("train_val_ratio"),
        noise_pct=config.get("mask_pct"),
        noise_stdev=config.get("mask_pct_stdev"),
        mask_val=config.get("mask_val"),
        feat_probs=config.get("mask_feat_probs"),
        shuffle=True,
        dtype=tf.float64,
        rand_seed=config.get("random_seed"),
        )

"""
Call the ModelDir with the configuration to make the
model directory and compile the model, then train it
using the generators above
"""

model,md = ModelDir.build_from_config(
        config,
        model_parent_dir=model_parent_dir,
        print_summary=True
        )
best_model = train(
        model_dir_path=md.dir,
        train_config=config,
        compiled_model=model,
        gen_training=gen_train,
        gen_validation=gen_val,
        )
