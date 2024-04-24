""" Methods for building and training neural networks """
import numpy as np
import json
from pathlib import Path
from random import random

import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Layer, Dense
from tensorflow.keras.layers import Conv2D,Lambda,Concatenate

from tracktrain.config import dense_kwargs_default

def get_dense_stack(name:str, layer_input:Layer, node_list:list,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Simple stack of dense layers with optional dropout and batchnorm
    """
    dense_kwargs = {**dense_kwargs_default.copy(), **dense_kwargs}
    l_prev = layer_input
    for i in range(len(node_list)):
        l_new = Dense(
                units=node_list[i],
                **dense_kwargs,
                name=f"{name}_dense_{i}"
                )(l_prev)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

def kl_divergence(z_mean, z_log_var):
    """ """
    return tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            ) * -0.5

def get_vi_projection(name:str, layer_input, num_latent:int):
    """
    Use a single feedforward layer to project the input to a num_latent
    dimensional mean and log variance parameters of the latent distributions.

    Return the parameters, and the latent vector sampled from the distribution.

    :@param layer_input: Input tensorflow Layer
    :@param num_latent: Number of latent distributions sampled, and as such
        the size of this layer's output.

    :@param return: 3-tuple (sample,z_mean,z_log_var) containing the sample,
        and the latent distributions' means and standard deviations.
    """
    ## Project to requested latent dimension for predicted mean and variance
    z_mean = Dense(num_latent, name=f"{name}_mean",
                   activation="linear")(layer_input)
    z_log_var = Dense(num_latent, name=f"{name}_logvar",
                      activation="linear")(layer_input)
    ## Draw a sample from the distribution from predicted parameters
    ## shaped like (B,L) for B batch elements and L latent features.
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    ## Concatenate the sample values to a (B,V,L) shape for B elements in the
    ## batch, V variational features (sample,z_mean,z_log_var), and L latent.
    return (sample,z_mean,z_log_var)

'''
class KL_Divergence(Layer):
    def call(self,inputs):
        z_sample,z_mean,z_log_var = inputs
        ## Calculate the KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
                )
        self.add_loss(kl_loss)
'''

def variational_encoder_decoder(
        name:str, num_inputs:int, num_outputs:int, num_latent:int,
        enc_node_list, dec_node_list, dropout_rate=0.0, batchnorm=True,
        softmax_out=False, enc_dense_kwargs={}, dec_dense_kwargs={}):
    """
    It's probably better to use the VariationalEncoderDecoder class.

    Construct a variational encoder decoder model parameterized by num_latent
    gaussian distributions approximating the latent posterior p(z|x) of input x

    :@param name: Model name for layer labeling
    :@param num_inputs: Number of inputs to the model
    :@param num_outputs: Number of values predicted by the model
    :@param num_latent: Number of latent distributions.
    :@param enc_node_list: List of int node counts per layer in the encoder
    :@param dec_node_list: List of int node counts per layer in the decoder
    :@param dropout_rate: Dropout rate in [0,1]
    :@param batchnorm: if True, do batchnorm regularization
    :@param enc_dense_kwargs: arguments (like activation function) passed to
        all of the encoder feedforward layers.
    :@param dec_dense_kwargs: arguments (like activation function) passed to
        all of the decoder feedforward layers.
    :@param softmax_out: if True, uses a softmax rather than linear layer as
        the output, for use in classifiers
    """
    l_input = Input(shape=(num_inputs,), name=f"{name}_input")
    l_enc_dense = get_dense_stack(
            name=f"{name}_enc",
            node_list=enc_node_list,
            layer_input=l_input,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=enc_dense_kwargs,
            )
    sample,z_mean,z_log_var = get_vi_projection(
            name=name,
            layer_input=l_enc_dense,
            num_latent=num_latent,
            )
    l_decoder = get_dense_stack(
            name=f"{name}_dec",
            node_list=dec_node_list,
            layer_input=sample,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dec_dense_kwargs,
            )
    l_output = Dense(
            num_outputs,
            name=f"{name}_out",
            activation="linear",
            )(l_decoder)
    if softmax_out:
        tf.keras.activations.softmax(l_output)
    ved = Model(l_input, l_output)
    ved.add_loss(kl_divergence(z_mean, z_log_var))
    return ved

def _apply_psf(args):
    """ Apply the psf (which should sum to 1) along the 2nd and 3rd axes """
    return tf.math.reduce_sum(tf.math.multiply(*args), axis=[1,2])

def get_paed(
        num_modis_bands:int, num_geom_bands:int,
        num_latent_bands:int, num_ceres_bands:int,
        enc_conv_filters:list, dec_conv_filters:list,
        enc_activation="gelu", enc_use_bias=True,
        dec_activation="gelu", dec_use_bias=True,
        enc_kwargs={}, enc_out_kwargs={}, enc_dropout=0., enc_batchnorm=True,
        dec_kwargs={}, dec_out_kwargs={}, dec_dropout=0., dec_batchnorm=True,
        **kwargs):
    """
    Pixel-wise aggregate encoder-decoder for aes690final
    Inputs:dict
        "modis":(B,M,N,Fm) grid of normalized Fm MODIS radiances
        "geom":(B,M,N,Fg) grid of normalized Fg geometry values
        "psf":(B,M,N,1) grid of PSF magnitudes summing to 1.
    Outputs:
        (B,Fc) normalized Fc CERES fluxes
    """
    #p_in = Input(shape=(grid_size,grid_size,1), name="in_psf")
    #m_in = Input(shape=(grid_size,grid_size,num_modis_bands), name="in_modis")
    m_in = Input(shape=(None,None,num_modis_bands), name="in_modis")
    g_in = Input(shape=(None,None,num_geom_bands), name="in_geom")
    #geom_shape = (*tf.shape(m_in)[:3], num_geom_bands)
    #grid_geom = tf.broadcast_to(g_in, geom_shape, name="resize_geom")
    p_in = Input(shape=(None,None,1), name="in_psf")

    last_layer = m_in
    for i,n in enumerate(enc_conv_filters):
        last_layer = Conv2D(
                filters=n,
                kernel_size=1,
                activation=enc_activation,
                use_bias=enc_use_bias,
                name=f"enc_conv_{i}",
                )(last_layer)
        if enc_batchnorm:
            last_layer = BatchNormalization(name=f"enc_bn_{i}")(last_layer)
        if enc_dropout>0.:
            last_layer = Dropout(enc_dropout, name=f"enc_do_{i}")(last_layer)
    enc_out = Conv2D(
            filters=num_latent_bands,
            kernel_size=1,
            activation="linear",
            name="enc_out",
            **enc_out_kwargs,
            )(last_layer)

    last_layer = Concatenate(axis=-1, name="concat_geom")([enc_out, g_in])
    for i,n in enumerate(dec_conv_filters):
        last_layer = Conv2D(
                filters=n,
                kernel_size=1,
                activation=dec_activation,
                use_bias=dec_use_bias,
                name=f"dec_conv_{i}",
                )(last_layer)
        if dec_batchnorm:
            last_layer = BatchNormalization(name=f"dec_bn_{i}")(last_layer)
        if dec_dropout>0.:
            last_layer = Dropout(dec_dropout, name=f"dec_do_{i}")(last_layer)
    dec_out = Conv2D(
            filters=num_ceres_bands,
            kernel_size=1,
            activation="linear",
            name="dec_out",
            **dec_out_kwargs,
            )(last_layer)
    psf_out = Lambda(function=_apply_psf, name="psf")((dec_out, p_in))
    inputs = {"modis":m_in, "geom":g_in, "psf":p_in}
    return Model(inputs=inputs, outputs=[psf_out])

def feedforward_from_config(config:dict):
    """ Just a wrapper for dumping a dictionary into the constructor """
    return feedforward(**config)

def feedforward(
        model_name:str, node_list:list, num_inputs:int, num_outputs:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}, softmax_out=False,
        **kwargs):
    """
    Just a series of dense layers with some optional parameters

    :@param model_name: string name uniquely identifying the model
    :@param node_list: List representing the node count of each Dense layer
    :@param num_inputs: Number of features in the input layer
    :@param num_outputs: Number of predictions in the output layer
    :@param batchnorm: Normalizes layer activations to gaussian between layers
    :@param dropout_rate: Percentage of nodes randomly disabled during training
    :@param dense_kwargs: Dict of arguments to pass to all Dense layers.
    :@param softmax_out: if True, uses a softmax rather than linear layer as
        the output, for use in classifiers
    """
    ff_in = Input(shape=(num_inputs,), name="input")
    dense = get_dense_stack(
            name=model_name,
            node_list=node_list,
            layer_input=ff_in,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dense_kwargs,
            )
    output = Dense(
            units=num_outputs,
            activation="linear",
            name="output"
            )(dense)
    if softmax_out:
        output = tf.keras.activations.softmax(output)

    model = Model(inputs=ff_in, outputs=output)
    return model

def gen_noisy(X, Y, noise_pct=0, noise_stdev=0, mask_val=9999.,
              feat_probs:np.array=None, shuffle=True, rand_seed=None):
    """
    Generates (X, Y, sample_weight) triplets for training a model with maskable
    feature values. The percentage of masked values determines the weight.

    The feature dimension is always assumed to be the last one. For example...

    (S,F) is a dataset with F features measured in S samples
    (S,L,F) is a dataset with S sequences of L members, each having F features.
    (S,M,N,F) is a 2d (M,N) dataset with F features and S sample instances.

    :@param X: (S,F) array of S samples of F features.
    :@param Y: (S,P) array of S samples of P predicted values.
    :@param noise_pct: mean percentage of feature values to mask.
    :@param noise_stdev: standard deviation of feature values to mask.
    :@param mask_val: Value to substitute for the mask.
    :@param feat_probs: Array with the same size as number of features, which
        provides the relative probabilities of each feature being selected to
        be masked. Uniform by default.
    :@param shuffle: if True, randomly shuffles samples along the first axis.
    """
    num_feats = X.shape[-1]
    num_samples = X.shape[0]
    ## Make sure the feature probability array is formatted correctly
    if feat_probs is None:
        feat_probs = np.full(shape=(num_feats,), fill_value=1.)
    else:
        assert np.array(feat_probs).squeeze().shape == (num_feats,)
        assert np.all(feat_probs >= 0.)
    ## Shuffle along the sample axis, if requested.
    if shuffle:
        rand_idxs = np.arange(num_samples)
        np.random.seed(rand_seed)
        np.random.shuffle(rand_idxs)
        X = X[rand_idxs]
        Y = Y[rand_idxs]
    ## Preserve ratios and convert to probabilities summing to 1
    feat_probs = feat_probs / np.sum(feat_probs)
    ## Pick a number of features to mask according to a distribution of
    ## percentages saturating at 0 and 1, parameterized by the user.
    noise_dist = np.random.normal(noise_pct,noise_stdev,size=num_samples)
    mask_count = np.rint(np.clip(noise_dist,0,1)*num_feats).astype(int)
    feat_idxs = np.arange(num_feats).astype(int)
    ##(!!!) The feature dimension is always assumed to be the final one (!!!)##
    for i in range(num_samples):
        ## Choose indeces that will be masked in each sample
        mask_idxs = np.random.choice(
                feat_idxs,
                size=mask_count[i],
                replace=False,
                #p=feat_probs,
                )
        X[i,...,mask_idxs] = mask_val
    weights = 1-mask_count/num_feats
    for i in range(num_samples):
        yield (X[i], Y[i], weights[i])

def array_to_noisy_tv_gen(
        X, Y, tv_ratio=.8, noise_pct=0, noise_stdev=0, mask_val=9999.,
        feat_probs:np.array=None, shuffle=True, rand_seed=None,
        dtype=tf.float64):
    """
    Get training and validation dataset generators returning  3-tuples (x,y,w)
    for input x, true output y, and sample weight w. Optionally use a random
    masking strategy to corrupt a subset of the features, adjusting the sample
    weight proportional to the percentage of values that were masked.

    :@param X: Inputs as an array with shape like (S, ... ,F)
        for S samples and F input features. May be 2+ dimensional.
    :@param Y: Truth outputs as an array with shape like (S, ... ,P)
        for S samples and P predicted features. May be 2+ dimensional.
    :@param tv_ratio: Ratio of training samples to total (sans validation)
    :@param noise_pct: mean percentage of feature values to mask.
    :@param noise_stdev: standard deviation of feature values to mask.
    :@param mask_val: Value to substitute for the mask.
    :@param feat_probs: Array with the same size as number of features, which
        provides the relative probabilities of each feature being selected to
        be masked. Uniform by default.
    :@param shuffle: if True, randomly shuffles samples along the first axis.

    :@return: 2-tuple (training_generator, validation_generator) of
        tf.data.Dataset generators which can be functionally iterated on.
    """
    num_samples = X.shape[0]
    num_feats = X.shape[-1]
    assert Y.shape[0] == num_samples
    ## Shuffle the samples if requested
    if shuffle:
        rand_idxs = np.arange(num_samples)
        np.random.seed(rand_seed)
        np.random.shuffle(rand_idxs)
        X = X[rand_idxs]
        Y = Y[rand_idxs]
    ## split the samples into training and validation sets
    split_idx = np.array([int(tv_ratio*num_samples)])
    Tx,Vx = np.split(X, split_idx)
    Ty,Vy = np.split(Y, split_idx)
    ## Establish the generator output signature
    out_sig = (
            tf.TensorSpec(shape=Tx.shape[1:],dtype=dtype),
            tf.TensorSpec(shape=Ty.shape[1:], dtype=dtype),
            tf.TensorSpec(shape=tuple(), dtype=dtype),
            )
    ## Init the training and validation Datasets with gen_noisy generators
    if feat_probs is None:
        feat_probs = np.full(shape=(num_feats,), fill_value=1.)
    gen_train = tf.data.Dataset.from_generator(
            gen_noisy,
            args=(Tx,Ty,noise_pct,noise_stdev,mask_val, feat_probs, shuffle),
            output_signature=out_sig,
            )
    gen_val = tf.data.Dataset.from_generator(
            gen_noisy,
            args=(Vx,Vy,noise_pct,noise_stdev,mask_val, feat_probs, shuffle),
            output_signature=out_sig,
            )
    ## return tf.data.Dataset generators as a 2-tuple (training, validation)
    return gen_train,gen_val

if __name__=="__main__":
    """ Test for noisy generator function """
    input_shape = (64,9)
    output_shape = (64,2)

    array_in = np.sqrt(np.sum(np.swapaxes(np.stack(np.meshgrid(
        *[np.linspace(0,1,num=ax)**2 for ax in input_shape]
        )), 1, 2), axis=0))
    array_out = np.sqrt(np.sum(np.swapaxes(np.stack(np.meshgrid(
        *[np.linspace(0,1,num=ax)**2 for ax in output_shape]
        )), 1, 2), axis=0))

    gt,gv = array_to_noisy_tv_gen(
            X=array_in,
            Y=array_out,
            tv_ratio=.5,
            noise_pct=.1,
            noise_stdev=.2,
            mask_val=99999.,
            feat_probs=None,
            shuffle=False,
            )

    for x,y,w in gt.batch(4):
        print("\n")
        for i in range(x.shape[0]):
            print(f"(% unmasked: {w[i]:.2f}):",
                  [f": {v:.2f}" for v in list(x[i])]
                  )
            pass
