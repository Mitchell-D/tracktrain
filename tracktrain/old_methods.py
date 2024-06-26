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

class SquareRegLayerOld(Layer):
    """
    adds loss proportional to the absolute value of the inputs,
    scaled by a coefficient L(x) = square_coeff * x**2
    """
    def __init__(self, square_coeff=.25, **kwargs):
        super().__init__(**kwargs)
        assert square_coeff>0, "Negative loss not allowed"
        self._square_coeff = square_coeff
    def call(self, x):
        reg_loss = tf.math.reduce_mean(.25*x**2)
        self.add_loss(reg_loss)
        return x

def _apply_psf(args):
    """ Apply the psf (which should sum to 1) along the 2nd and 3rd axes """
    return tf.math.reduce_sum(tf.math.multiply(*args), axis=[1,2])

def get_paed_old(
        num_modis_feats:int, num_ceres_feats:int,
        num_latent_feats:int, num_ceres_labels:int,
        enc_conv_filters:list, dec_conv_filters:list,
        enc_activation="gelu", enc_use_bias=True,
        dec_activation="gelu", dec_use_bias=True,
        enc_kwargs={}, enc_out_kwargs={}, enc_dropout=0., enc_batchnorm=True,
        dec_kwargs={}, dec_out_kwargs={}, dec_dropout=0., dec_batchnorm=True,
        **kwargs):
    """
    Pixel-wise aggregate encoder-decoder for aes690final

    This one decodes the latent space before aggregating it.
    For a model that also aggregates the latent space before decoding,
    which is an experimental regularization method, see get_paed.


    Inputs: (dict)
        "modis":(B,M,N,Fm) grid of normalized Fm MODIS radiances
        "geom":(B,M,N,Fg) grid of normalized Fg geometry values
        "psf":(B,M,N,1) grid of PSF magnitudes summing to 1.
    Outputs:
        (B,Fc) normalized Fc CERES fluxes
    """
    m_in = Input(shape=(None,None,num_modis_feats), name="in_modis")
    g_in = Input(shape=(None,None,num_ceres_feats), name="in_geom")
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
            filters=num_latent_feats,
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
            filters=num_ceres_labels,
            kernel_size=1,
            activation="linear",
            name="dec_out",
            **dec_out_kwargs,
            )(last_layer)
    psf_out = Lambda(function=_apply_psf, name="psf")((dec_out, p_in))
    inputs = [m_in, g_in, p_in]
    return Model(inputs=inputs, outputs=[psf_out])

def get_paed_old2(
        num_modis_feats:int, num_ceres_feats:int,
        num_latent_feats:int, num_ceres_labels:int,
        enc_conv_filters:list, dec_conv_filters:list,
        enc_activation="gelu", enc_use_bias=True,
        dec_activation="gelu", dec_use_bias=True,
        enc_kwargs={}, enc_out_kwargs={}, enc_dropout=0., enc_batchnorm=True,
        dec_kwargs={}, dec_out_kwargs={}, dec_dropout=0., dec_batchnorm=True,
        kernel_size=1, **kwargs):
    """
    Pixel-wise aggregate encoder-decoder for aes690final.
    This model is essentially the same as get_paed, but is less expressive.
    It's just sticking around for now so I can generate figures for test-7

    Inputs: (dict)
        "modis":(B,M,N,Fm) grid of normalized Fm MODIS radiances
        "geom":(B,M,N,Fg) grid of normalized Fg geometry values
        "psf":(B,M,N,1) grid of PSF magnitudes summing to 1.
    Outputs:
        (B,Fc) normalized Fc CERES fluxes
    """
    m_in = Input(shape=(None,None,num_modis_feats), name="in_modis")
    g_in = Input(shape=(None,None,num_ceres_feats), name="in_geom")
    p_in = Input(shape=(None,None,1), name="in_psf")
    inputs = [m_in, g_in, p_in]

    def _get_encoder(enc_in):
        last_layer = enc_in
        for i,n in enumerate(enc_conv_filters):
            last_layer = Conv2D(
                    filters=n,
                    kernel_size=kernel_size,
                    activation=enc_activation,
                    use_bias=enc_use_bias,
                    padding="same", ## Always pad to same size even big kernels
                    name=f"enc_conv_{i}",
                    )(last_layer)
            if enc_batchnorm:
                last_layer = BatchNormalization(
                        name=f"enc_bn_{i}"
                        )(last_layer)
            if enc_dropout>0.:
                last_layer = Dropout(
                        enc_dropout,
                        name=f"enc_do_{i}"
                        )(last_layer)
        enc_out = Conv2D(
                filters=num_latent_feats,
                kernel_size=1,
                activation="linear",
                name="enc_out",
                **enc_out_kwargs,
                )(last_layer)
        return enc_out

    def _get_decoder(latent, geom, dec_str):
        #last_layer = tf.concat([latent,geom], -1)
        last_layer = Concatenate(
                axis=-1,
                name=f"concat_geom_{dec_str}",
                )([latent, geom])
        for i,n in enumerate(dec_conv_filters):
            last_layer = Conv2D(
                    filters=n,
                    kernel_size=kernel_size,
                    activation=dec_activation,
                    use_bias=dec_use_bias,
                    padding="same", ## Always pad to same size even big kernels
                    name=f"dec_conv_{dec_str}_{i}",
                    )(last_layer)
            if dec_batchnorm:
                last_layer = BatchNormalization(
                        name=f"dec_bn_{dec_str}_{i}"
                        )(last_layer)
            if dec_dropout>0.:
                last_layer = Dropout(
                        dec_dropout,
                        name=f"dec_do_{dec_str}_{i}"
                        )(last_layer)
        dec_out = Conv2D(
                filters=num_ceres_labels,
                kernel_size=1,
                activation="linear",
                name=f"dec_out_{dec_str}",
                **dec_out_kwargs,
                )(last_layer)
        return dec_out

    ## functional layer to apply the PSF to a (B,W,W,F) grid
    apply_psf = Lambda(function=_apply_psf, name="psf")

    ## encode the MODIS data to the latent grid
    enc_out = _get_encoder(m_in)

    ## decode the latent grid to a prediction, then aggregate it to a vector
    enc_dec = _get_decoder(enc_out, g_in, dec_str="dec-agg")
    enc_dec_agg = apply_psf((enc_dec, p_in))

    ## aggregate the latent grid to a vector, then decode it to a prediction
    enc_agg = apply_psf((enc_out, p_in))[:,tf.newaxis,tf.newaxis,:]
    g_agg = tf.math.reduce_mean(g_in, axis=(1,2), keepdims=True)
    enc_agg_dec = _get_decoder(enc_agg, g_agg, dec_str="agg-dec")

    ## average the outputs from the two different pathways
    flux = (enc_dec_agg + enc_agg_dec)/2

    return Model(inputs=inputs, outputs=[flux])

def get_paed(
        num_modis_feats:int, num_ceres_feats:int,
        num_latent_feats:int, num_ceres_labels:int,
        enc_conv_filters:list, dec_conv_filters:list,
        enc_activation="gelu", enc_use_bias=True,
        dec_activation="gelu", dec_use_bias=True,
        enc_kwargs={}, enc_out_kwargs={}, enc_dropout=0., enc_batchnorm=True,
        dec_kwargs={}, dec_out_kwargs={}, dec_dropout=0., dec_batchnorm=True,
        square_regularization_coeff=None, share_decoder_weights=True,
        kernel_size=1, separate_output_decoders=False, **kwargs):
    """
    Pixel-wise aggregate encoder-decoder for aes690final.

    Inputs: (dict)
        "modis":(B,M,N,Fm) grid of normalized Fm MODIS radiances
        "geom":(B,M,N,Fg) grid of normalized Fg geometry values
        "psf":(B,M,N,1) grid of PSF magnitudes summing to 1.
    Outputs:
        (B,Fc) normalized Fc CERES fluxes

    :@param square_regularization_coeff:If float instead of None, the
        predictions are regularized by penalizing distance from 0. This is
        done by squaring the outputs, then multiplying by this coefficient.
    :@param share_decoder_weights: If True, the same weights will be used to
        decode the aggregate latent vector and the full pixel-wise grid.
        Otherwise, each pathway will get its own decoder.
    :@param separate_output_decoders: If True, each output (num_ceres_labels)
        will be decoded by separate weights. If share_decoder_weights is also
        True, each prediction will be made using 2 separate decoders.
    """
    m_in = Input(shape=(None,None,num_modis_feats), name="in_modis")
    g_in = Input(shape=(None,None,num_ceres_feats), name="in_geom")
    p_in = Input(shape=(None,None,1), name="in_psf")
    inputs = [m_in, g_in, p_in]

    def _get_encoder(enc_in):
        last_layer = enc_in
        for i,n in enumerate(enc_conv_filters):
            last_layer = Conv2D(
                    filters=n,
                    kernel_size=kernel_size,
                    activation=enc_activation,
                    use_bias=enc_use_bias,
                    padding="same", ## Always pad to same size even big kernels
                    name=f"enc_conv_{i}",
                    )(last_layer)
            if enc_batchnorm:
                last_layer = BatchNormalization(
                        name=f"enc_bn_{i}"
                        )(last_layer)
            if enc_dropout>0.:
                last_layer = Dropout(
                        enc_dropout,
                        name=f"enc_do_{i}"
                        )(last_layer)
        enc_out = Conv2D(
                filters=num_latent_feats,
                kernel_size=1,
                activation="linear",
                name="enc_out",
                **enc_out_kwargs,
                )(last_layer)
        return enc_out

    #'''
    def _get_new_decoder(latent, geom, dec_str, output_count=None):
        """
        Returns new decoder declaration with the provided inputs.
        Using this method creates a new set of weights.

        :@param latent: (B,W,W,L) latent grid
        :@param geom: (B,W,W,G) grid of CERES features
        :@param dec_str: identifying string for this decoder
        :@param output_count: Number of elements in the output.
            If None, defaults to num_ceres_labels
        """
        #last_layer = tf.concat([latent,geom], -1)
        last_layer = Concatenate(
                axis=-1,
                name=f"concat_geom_{dec_str}",
                )([latent, geom])
        for i,n in enumerate(dec_conv_filters):
            last_layer = Conv2D(
                    filters=n,
                    kernel_size=kernel_size,
                    activation=dec_activation,
                    use_bias=dec_use_bias,
                    padding="same", ## Always pad to same size even big kernels
                    name=f"dec_conv_{dec_str}_{i}",
                    )(last_layer)
            if dec_batchnorm:
                last_layer = BatchNormalization(
                        name=f"dec_bn_{dec_str}_{i}"
                        )(last_layer)
            if dec_dropout>0.:
                last_layer = Dropout(
                        dec_dropout,
                        name=f"dec_do_{dec_str}_{i}"
                        )(last_layer)
        num_out = num_ceres_labels if output_count is None else output_count
        dec_out = Conv2D(
                filters=num_out,
                kernel_size=1,
                activation="linear",
                name=f"dec_out_{dec_str}",
                **dec_out_kwargs,
                )(last_layer)
        return dec_out
    #'''

    def _get_model_decoder(dec_str, output_count=None):
        """
        Returns functional decoder (layers not applied; weights shared)

        :@param dec_str: identifying string for this decoder
        :@param output_count: Number of elements in the output.
            If None, defaults to num_ceres_labels
        """
        dec_latent_in = Input(
                shape=(None,None,num_latent_feats),
                name="dec_in_latent")
        dec_geom_in = Input(
                shape=(None,None,num_ceres_feats),
                name="dec_in_geom")
        last_layer = Concatenate(
                axis=-1,
                name=f"concat_geom",
                )([dec_latent_in,dec_geom_in])
        for i,n in enumerate(dec_conv_filters):
            last_layer = Conv2D(
                    filters=n,
                    kernel_size=kernel_size,
                    activation=dec_activation,
                    use_bias=dec_use_bias,
                    padding="same", ## Always pad to same size even big kernels
                    name=f"dec_conv_{i}",
                    )(last_layer)
            if dec_batchnorm:
                last_layer = BatchNormalization(
                        name=f"dec_bn_{i}"
                        )(last_layer)
            if dec_dropout>0.:
                last_layer = Dropout(
                        dec_dropout,
                        name=f"dec_do_{i}"
                        )(last_layer)
        num_out = num_ceres_labels if output_count is None else output_count
        dec_out = Conv2D(
                filters=num_out,
                kernel_size=1,
                activation="linear",
                name=f"dec_out_{dec_str}",
                **dec_out_kwargs,
                )(last_layer)
        dec = Model(inputs=[dec_latent_in,dec_geom_in],
                    outputs=dec_out, name=f"dec_{dec_str}")
        return dec

    ## functional layer to apply the PSF to a (B,W,W,F) grid
    apply_psf = Lambda(function=_apply_psf, name="psf")

    ## encode the MODIS data to the latent grid
    latent_grid = _get_encoder(m_in)


    if separate_output_decoders:
        """ Separate output decoders - full latent grid decoding  """
        if share_decoder_weights:
            ## declare a shared latent grid decoder for each output
            decoders = [_get_model_decoder(f"dec-agg-{i}", output_count=1)
                        for i in range(num_ceres_labels)]
            enc_dec = [d([latent_grid, g_in]) for d in decoders]
        else:
            ## declare separate latent grid decoders for each output
            enc_dec = [
                    _get_new_decoder(
                        latent_grid, g_in, dec_str=f"dec-agg-{i}",
                        output_count=1,)
                    for i in range(num_ceres_labels)
                    ]
        ## Optionally regularize un-aggregated outputs by their magnitude to
        ## dissuade the model from over-representing individual pixels
        if not square_regularization_coeff is None:
            weight_reg = SquareRegLayerOld(square_regularization_coeff,
                                           name="square_reg")
            enc_dec = [weight_reg(d) for d in enc_dec]
        ## Apply the point spread function to the decoded latent grid
        enc_dec_agg = [apply_psf((d, p_in)) for d in enc_dec]
        enc_dec_agg = Concatenate(axis=-1, name="join_dec-agg")(enc_dec_agg)

        """ Separate output decoders - aggregate latent vector decoding  """
        ## aggregate the latent grid to a vector, then decode to a prediction
        latent_agg = apply_psf((latent_grid, p_in))[:,tf.newaxis,tf.newaxis,:]
        ## average the geometry (which should be constant during training)
        geom_agg = tf.math.reduce_mean(g_in, axis=(1,2), keepdims=True)
        if share_decoder_weights:
            enc_agg_dec = [d([latent_agg, geom_agg])
                           for d in decoders]
        else:
            ## Declare a new aggregated latent vector decoder for each output
            enc_agg_dec = [
                    _get_new_decoder(
                        latent_agg, geom_agg, dec_str=f"agg-dec-{i}",
                        output_count=1,)
                    for i in range(num_ceres_labels)
                    ]
        enc_agg_dec = Concatenate(axis=-1, name="join_agg-dec")(enc_agg_dec)
    else:
        """ Single output decoder - full latent grid decoding  """
        ## decode the latent grid to a prediction, then aggregate to a vector
        if share_decoder_weights:
            decoder = _get_model_decoder("dec-agg")
            enc_dec = decoder([latent_grid, g_in])
        else:
            enc_dec = _get_new_decoder(latent_grid, g_in, dec_str="dec-agg")

        ## Optionally regularize un-aggregated outputs by their magnitude to
        ## dissuade the model from over-representing individual pixels
        if not square_regularization_coeff is None:
            weight_reg = SquareRegLayerOld(square_regularization_coeff)
            enc_dec = weight_reg(enc_dec)

        enc_dec_agg = apply_psf((enc_dec, p_in))

        """ Single output decoder - aggregate latent vector decoding  """
        ## aggregate the latent grid to a vector, then decode it to a prediction
        latent_agg = apply_psf((latent_grid, p_in))[:,tf.newaxis,tf.newaxis,:]
        ## average the geometry (which should be constant during training)
        geom_agg = tf.math.reduce_mean(g_in, axis=(1,2), keepdims=True)
        print()
        print(enc_dec_agg.shape)
        print()
        if share_decoder_weights:
            enc_agg_dec = decoder([latent_agg, geom_agg])
        else:
            enc_agg_dec = _get_new_decoder(
                    latent_agg,
                    geom_agg,
                    dec_str="agg-dec"
                    )

    ## average the outputs from the two different pathways
    flux = (enc_dec_agg + enc_agg_dec)/2

    return Model(inputs=inputs, outputs=[flux])

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
