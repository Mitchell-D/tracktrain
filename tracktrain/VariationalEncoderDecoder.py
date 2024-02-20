import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
from tensorflow.keras.saving import register_keras_serializable

import tracktrain.model_methods as mm
from tracktrain.utils import validate_keys
from tracktrain.config import vae_valid_args, vae_arg_descriptions
from tracktrain.config import vae_arg_defaults


@register_keras_serializable(package="variational")
class VariationalEncoderDecoder(Model):
    """
    Class implementing tf.keras.Model for a variational encoder-decoder with
    the following architecture:

    (1) A series of Dense layers for the encoder.
    (2) A num_latent-dimensional gaussian parameterization of the mean and
        standard deviation of the latent posterior P(z|x).
    (3) A routine for sampling a latent vector z from the latent distriubtions.
    (4) A series of Dense layers for the decoder.
    (5) Output a num_inputs value
    """
    @staticmethod
    @register_keras_serializable(package="variational")
    def from_config(config):
        """
        initialize a VariationalEncoderDecoder from a configuration file that
        minimally has
        """
        ## Add defaults for any missing arguments; user config gets precedent.
        config = {**vae_arg_defaults, **config}
        validate_keys(
            mandatory_keys=vae_valid_args,
            received_keys=list(config.keys()),
            source_name="VariationalEncoderDecoder",
            descriptions=vae_arg_descriptions,
            )
        return VariationalEncoderDecoder(**config)

    @staticmethod
    @register_keras_serializable(package="variational")
    def sample(mean, log_var):
        """
        The reparameterization trick. Sample a value from a normal
        distribution given the encoder's mean and log variance.

        See this blog post:
        https://gregorygundersen.com/blog/2018/04/29/reparameterization/

        :@param mean: (batch,num_latent) input tensor for mean parameter
        :@param log_var: (batch,num_latent) input tensor for log variance
        """
        assert mean.shape == log_var.shape
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    @staticmethod
    @register_keras_serializable(package="variational")
    def kl_loss(mean, log_var):
        """
        Calculate the KL divergence loss

        :@param mean: Output from mean layer (B,L)
        :@param log_var: Output from log variance layer (B,L)

        :@return: 2-tuple (recon_loss, kl_loss)
        """
        ## Taken from (Kingma & Welling, 2014) Equation 10
        kl_loss = -1/2 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var)
                )
        return kl_loss

    def __init__(
            self, model_name:str, num_inputs:int, num_outputs:int,
            num_latent:int, enc_node_list:list, dec_node_list:list,
            batchnorm:bool=True, dropout_rate:float=0.0,
            enc_dense_kwargs={}, dec_dense_kwargs={}, *args, **kwargs):
        """
        Initialize a variational encoder-decoder model, consisting of a
        sequence of feedforward layers encoding a num_latent dimensional
        distribution.

        :@param model_name: String name of this model (for naming output files)
        :@param num_inputs: Number of inputs received by the model.
        :@param num_outputs: Number of outputs predicted by the model
        :@param num_latent: Dimensionality of the latent distribution.
        :@param enc_node_list: List of ints corresponding to the width of each
            encoder layer in number of nodes.
        :@param dec_node_list:List of ints corresponding to the width of each
            decoder layer in number of nodes.
        :@param dropout_rate: Ratio of random nodes disabled during training
        :@param batchnorm: If True, normalizes layer-wise activation amounts.
        :@param enc_dense_kwargs: dict of args to init encoder Dense layers.
        :@param dec_dense_kwargs: dict of args to init decoder Dense layers.
        """
        super(VariationalEncoderDecoder, self).__init__(self)
        self.model_name = model_name
        self.num_latent = num_latent
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        """ Initialize encoder layers """
        self.encoder_input = Input(shape=self.num_inputs)
        self.encoder_layers = mm.get_dense_stack(
                name=f"{self.model_name}-enc",
                layer_input=self.encoder_input,
                node_list=enc_node_list,
                batchnorm=batchnorm,
                dropout_rate=dropout_rate,
                dense_kwargs=enc_dense_kwargs,
                )
        self.mean_layer = Dense(
                self.num_latent,
                name=f"{self.model_name}-zmean",
                )(self.encoder_layers)
        self.log_var_layer = Dense(
                self.num_latent,
                name=f"{self.model_name}-zlogvar",
                )(self.encoder_layers)

        """ Initialize decoder layers """
        self.decoder_input = Input(shape=(self.num_latent,))
        self.decoder_layers = mm.get_dense_stack(
                name=f"{self.model_name}-dec",
                layer_input=self.decoder_input,
                node_list=dec_node_list,
                batchnorm=batchnorm,
                dropout_rate=dropout_rate,
                dense_kwargs=dec_dense_kwargs,
                )
        self.decoder_output = Dense(
                self.num_outputs,
                activation="linear"
                )(self.decoder_layers)

        """ Make separate functional Models for the encoder and decoder """
        self.encoder = Model(
                inputs=self.encoder_input,
                outputs=(self.mean_layer, self.log_var_layer),
                )
        self.decoder = Model(
                inputs=self.decoder_input,
                outputs=self.decoder_output,
                )
        self.build(input_shape=(None,num_inputs,))

    def call(self, inputs, return_params=False):
        """
        Call the full model (encoder and decoder) on a (B,F) tensor of
        B batch samples each having F inputs, and return the result.

        :@param inputs: (B,F) tensor of B input vectors each with F features
        :@param return_params: If True, returns the latent distribution params
            alongside the output vector as a 3-tuple like (Y, mean, log_var)
        """
        X,_,_ = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        mean, log_var = self.encoder(X)
        Z = self.sample(mean, log_var)
        Y = self.decoder(Z)
        if return_params:
            return (Y, mean, log_var)
        return Y

    def train_step(self, data):
        """
        Call the model while running the forward pass and calculating loss
        in a GradientTape context, and uptimize the weights by applying the
        gradients.

        :@param data: (B,F) tensor for B batch samples of F features

        :@return: dict {"loss":float} characterizing both the KL divergence
            and the reconstruction loss from the training step.
        """
        ## inputs,outputs,sample weights
        X,Y,W = tf.keras.utils.unpack_x_y_sample_weight(data)
        ## Call this model within the gradient-recorded context
        with tf.GradientTape() as tape:
            ## Run the loss function
            mean,log_var = self.encoder(X)
            Z = self.sample(mean,log_var)
            P = self.decoder(Z)
            kl_loss = VariationalEncoderDecoder.kl_loss(mean, log_var)
            rec_loss = MeanSquaredError()(P,Y)
            ## Get the gradients from the recorded training step
            ## trainable_variables is an attribute exposed by parent Model
            W = (W, 1.)[W is None]
            loss = tf.cast(W,tf.float32) * \
                    (tf.cast(rec_loss,tf.float32) + kl_loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                    zip(gradients,self.trainable_variables))
        self.compiled_metrics.update_state(Y, P, W)
        return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer, loss, metrics, weighted_metrics=None,
                *args, **kwargs):
        """
        Override the compile method to capture user arguments

        :@param optimizer: String label or keras.optimizers object
        :@param loss_fn: String label or serializable loss function to
            evaluate reconstruction accuracy. KL divergence loss is applied
            regardless of this value.
        :@param metrics: List of metrics or keras.metrics object
        """
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics,
                        weighted_metrics=weighted_metrics, *args, **kwargs)
        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.deserialize(loss)
        self.loss_metrics = metrics
        return self

    def get_compile_config(self):
        """
        Return parameters that need to be serialized to save
        """
        return {
            "optimizer":self.optimizer,
            "loss":tf.keras.losses.serialize(self.loss_fn),
            "metrics":self.loss_metrics,
            }

    def compile_from_config(self, config):
        # Deserializes the compile parameters (important, since many are custom)
        optimizer = config.get("optimizer")
        loss_fn = tf.keras.losses.deserialize(config.get("loss"))
        metrics = config.get("metrics")
        # Calls compile with the deserialized parameters
        self.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

if __name__=="__main__":
    """ Unit test for encoder-decoder """
    ved = VariationalEncoderDecoder(
            model_name="test",
            num_inputs=8,
            num_outputs=2,
            num_latent=8,
            enc_node_list=[64,64,32,16],
            dec_node_list=[16,16],
            )
    def myloss(y,p):
        return (y-p)**2
    ved.compile(
            optimizer="adam",
            loss=myloss,
            metrics=["mse", "mae"],
            )
    cfg = ved.get_compile_config()
    ved.compile_from_config(cfg)
    ved.compile_from_config({"optimizer":"sgd", "loss":"mae", "metrics":"mse"})
