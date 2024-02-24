"""
Script for randomly searching a user-defined combinatorial graph
of model configurations within the tracktrain framework.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

wrf_nc_path = Path("wrfvars.2018-07-21_11-00-00.nc")

#input_feats = ["tmpc","dwpc","relh","sknt", "mslp","p01m","gust","feel"]


"""
base_config contains configuration values to all models,
so it should only have fields not subject to variations.
"""
base_config = {
        ## Meta-info
        #"model_name":"test-15",
        "data_source":wrf_nc_path.as_posix(),
        "model_type":"ved",
        "rand_seed":20240128,

        ## Exclusive to feedforward
        #"node_list":[64,64,32,32,16],
        #"dense_kwargs":{"activation":"relu"},

        ## Exclusive to variational encoder-decoder
        #"num_latent":8,
        #"enc_node_list":[64,64,32,32,16],
        #"dec_node_list":[16,32,32,64],
        #"enc_dense_kwargs":{"activation":"relu"},
        #"dec_dense_kwargs":{"activation":"relu"},

        ## Common to models
        "batchnorm":True,
        #"dropout_rate":0.0,

        ## Exclusive to compile_and_build_dir
        #"learning_rate":1e-5,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":64,
        "batch_buffer":4,
        "max_epochs":2048, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        "train_val_ratio":.9,
        "mask_pct":0.0,
        "mask_pct_stdev":0.0,
        "mask_val":9999,
        "mask_feat_probs":None,

        "notes":"Variational model for predicting irrigation amount",
        }

"""
The variations dictionary maps config field names to tuples containing valid
values for that field. For each new model, one of the configurations in
the tuple corresponding to each field will be selected as that field's
parameter for the model run.

If many variations are specified, the combinatorial space will be too large
to fully search. In order to evaluate a wide variety of possible hyper-
-parameterizations, a random selection is made for each field at every run.
"""

num_samples = 32
model_base_name = "ved-irr"
variations = {
        "dropout_rate":(0.0,0.1,0.2,0.4),
        "learning_rate":(1e-6,1e-4,1e-2),
        "train_val_ratio":(.6,.8,.9),
        "mask_pct":(0,0,0,.1,),
        "mask_pct_stdev":(0,0,0,.2),

        ## FF only
        "node_list":(
            (32,32,16),
            (64,64,32,32,16),
            (64,64,64,32,32,32,16),
            (256,256,256,64,64,64,32,32,32,16),
            (16,32,32,32,64,64,64,256,256,256),
            (16,32,32,32,64,64,64),
            (16,32,32,64,64),
            (16,32,32),
            ),
        "dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),

        ## VED only
        "num_latent":(4,8,12),
        "enc_node_list":(
            (32,32,16),
            (64,64,32,32,16),
            (64,64,64,32,32,32,16),
            (256,256,256,64,64,64,32,32,32,16),
            ),
        "dec_node_list":(
            (16,32,32),
            (16,32,32,64,64),
            (16,32,32,32,64,64,64),
            (16,32,32,32,64,64,64,256,256,256),
            ),
        "enc_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),
        "dec_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            {"activation":"tanh"},
            ),
        }

vlabels,vdata = zip(*variations.items())

comb_failed = []
comb_trained = []
vdata = tuple(map(tuple, vdata))
comb_shape = tuple(len(v) for v in vdata)
comb_count = np.prod(np.array(comb_shape))
for i in range(num_samples):
    ## Get a random argument combination from the configuration
    cur_comb = tuple(np.random.randint(0,j) for j in comb_shape)
    cur_update = {
            vlabels[i]:vdata[i][cur_comb[i]]
            for i in range(len(vlabels))
            }
    cur_update["model_name"] = model_base_name+f"-{i:03}"
    cur_config = {**base_config, **cur_update}
    try:
        ## Build a config dict for the selected current combination

        ## Extract and preprocess the data
        from preprocess import load_WRFSCM_training_data
        X,Y,xlabels,ylabels,y_scales = load_WRFSCM_training_data(wrf_nc_path)

        cur_config["num_inputs"] = X.shape[-1]
        cur_config["num_outputs"] = Y.shape[-1]
        cur_config["input_feats"] = xlabels
        cur_config["output_feats"] = ylabels

        ## Initialize the masking data generators
        gen_train,gen_val = mm.array_to_noisy_tv_gen(
                X=X,
                Y=Y,
                tv_ratio=cur_config.get("train_val_ratio"),
                noise_pct=cur_config.get("mask_pct"),
                noise_stdev=cur_config.get("mask_pct_stdev"),
                mask_val=cur_config.get("mask_val"),
                feat_probs=cur_config.get("mask_feat_probs"),
                shuffle=True,
                dtype=tf.float64,
                rand_seed=cur_config.get("random_seed"),
                )
        ## Initialize the model
        model,md = ModelDir.build_from_config(
                cur_config,
                model_parent_dir=Path("models"),
                print_summary=False,
                )
        best_model = train(
            model_dir_path=md.dir,
            train_config=cur_config,
            compiled_model=model,
            gen_training=gen_train,
            gen_validation=gen_val,
            )
    except Exception as e:
        print(f"FAILED update combination {cur_update}")
        raise e
        #print(e)
        comb_failed.append(cur_comb)
    comb_trained.append(cur_comb)
