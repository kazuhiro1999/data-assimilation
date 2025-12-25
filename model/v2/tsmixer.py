'''
TSMixer: Time Series Mixer for Forecasting
https://github.com/ditschuk/pytorch-tsmixer
'''

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Model

from model.normalizing_flow import PlanarFlow, InvertibleFlow, NormalizingFlow

# Utility functions
def time_to_feature(x):
    return tf.transpose(x, perm=[0, 2, 1])

def feature_to_time(x):
    return tf.transpose(x, perm=[0, 2, 1])


# TimeBatchNorm2D equivalent in TensorFlow
class TimeBatchNorm(tf.keras.layers.Layer):
    def __init__(self, time_steps, channels):
        super().__init__()
        self.norm = layers.BatchNormalization()
        self.time_steps = time_steps
        self.channels = channels

    def call(self, x):
        x = tf.reshape(x, [-1, self.time_steps * self.channels])
        x = self.norm(x)
        return tf.reshape(x, [-1, self.time_steps, self.channels])


# Feature Mixing
class FeatureMixing(tf.keras.layers.Layer):
    def __init__(self, sequence_length, input_channels, output_channels, ff_dim, 
                 activation='relu', dropout_rate=0.1, normalize_before=True, norm_type='batch'):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.ff_dim = ff_dim
        self.norm_type = norm_type
        
        Norm = TimeBatchNorm if norm_type == 'batch' else layers.LayerNormalization
        self.norm_before = Norm(sequence_length, input_channels) if normalize_before else layers.Activation('linear')
        self.norm_after = Norm(sequence_length, output_channels) if not normalize_before else layers.Activation('linear')

        self.fc1 = layers.Dense(ff_dim)
        self.fc2 = layers.Dense(output_channels)
        self.activation = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout_rate)

        self.projection = layers.Dense(output_channels) if input_channels != output_channels else tf.identity

    def call(self, x):
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + x_proj
        x = self.norm_after(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'ff_dim': self.ff_dim,
            'norm_type': self.norm_type,
        })
        return config


# Time Mixing
class TimeMixing(tf.keras.layers.Layer):
    def __init__(self, sequence_length, input_channels, activation='relu', dropout_rate=0.1, norm_type='batch'):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.activation_fn = activation
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        
        Norm = TimeBatchNorm if norm_type == 'batch' else layers.LayerNormalization
        self.norm = Norm(sequence_length, input_channels)
        self.fc = layers.Dense(sequence_length)
        self.activation = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        x_temp = feature_to_time(x)
        x_temp = self.fc(x_temp)
        x_temp = self.activation(x_temp)
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)
        return self.norm(x + x_res)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'input_channels': self.input_channels,
            'activation': self.activation_fn,
            'dropout_rate': self.dropout_rate,
            'norm_type': self.norm_type,
        })
        return config


# Mixer Layer
class MixerLayer(tf.keras.layers.Layer):    
    
    def __init__(self, sequence_length, input_channels, output_channels, ff_dim, 
                 activation='relu', dropout_rate=0.1, normalize_before=True, norm_type='batch'):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.normalize_before = normalize_before
        self.norm_type = norm_type
        
        self.time_mixing = TimeMixing(sequence_length, input_channels, activation, dropout_rate, norm_type)
        self.feature_mixing = FeatureMixing(sequence_length, input_channels, self.output_channels, ff_dim, 
                                            activation, dropout_rate, normalize_before, norm_type)

    def call(self, x):
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x
    
    def compute_output_shape(self, input_shape):
        # Input shape: (batch_size, sequence_length, input_channels)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        # Output shape after time mixing and feature mixing will have the same sequence length and output_channels
        return (batch_size, sequence_length, self.feature_mixing.output_channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'ff_dim': self.ff_dim,
            'normalize_before': self.normalize_before,
            'norm_type': self.norm_type,
        })
        return config


# TSMixer Model
class TSMixer(tf.keras.Model):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """
        
    def __init__(self, sequence_length, prediction_length, input_channels, output_channels=None,
                 activation='relu', num_blocks=2, dropout_rate=0.1, ff_dim=64,
                 normalize_before=True, norm_type='batch', *args, **kwargs):
        super().__init__()
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_channels = input_channels
        self.output_channels = output_channels or input_channels
        self.activation = activation
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.normalize_before = normalize_before
        self.norm_type = norm_type
        
        self.mixer_layers = [
            MixerLayer(sequence_length, input_channels if i == 0 else self.output_channels,
                       self.output_channels, ff_dim, activation, dropout_rate, normalize_before, norm_type)
            for i in range(num_blocks)
        ]

        self.temporal_projection = layers.Dense(prediction_length)

    def call(self, x):
        for layer in self.mixer_layers:
            x = layer(x)
        x = feature_to_time(x)
        x = self.temporal_projection(x)
        x = time_to_feature(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'activation': self.activation,
            'num_blocks': self.num_blocks,
            'dropout_rate': self.dropout_rate,
            'ff_dim': self.ff_dim,
            'normalize_before': self.normalize_before,
            'norm_type': self.norm_type,
        })
        return config


# VariationalTSMixer Model
class VariationalTSMixer(TSMixer):
    def __init__(self, output_channels, *args, **kwargs):
        super(VariationalTSMixer, self).__init__(*args, **kwargs)
        
        self.final_mean_layer = layers.Dense(output_channels)
        self.final_logvar_layer = layers.Dense(output_channels)
    
    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))
    
    def decode_allsteps(self, x):
        for layer in self.mixer_layers:
            x = layer(x)

        x = feature_to_time(x)
        x = self.temporal_projection(x)
        x = time_to_feature(x)
        
        mean = self.final_mean_layer(x)
        logvar = self.final_logvar_layer(x)
        logvar = self.soft_clip_tanh(logvar, min_val=-6.0, max_val=3.0)
        return mean, logvar

    def call(self, x, training=False, return_sample=False):
        
        mean, logvar = self.decode_allsteps(x)

        if return_sample:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(tf.shape(std))
            sample = mean + eps * std  # Reparameterization trick
            x_out = sample
        else:
            x_out = mean

        return x_out, mean, logvar
    
    def get_config(self):
        config = super().get_config()
        return config
    
    
class VariationalTSMixerNF(VariationalTSMixer):
    def __init__(self, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.flow_layers = [
            PlanarFlow(latent_dim),
            InvertibleFlow(latent_dim),
            PlanarFlow(latent_dim),
        ]
        self.normalizing_flow = NormalizingFlow(self.flow_layers)

    def build(self, input_shape):
        super().build(input_shape)
        self.normalizing_flow.build((None, self.prediction_length, self.latent_dim))  # shape: (batch, dim)

    def sample_latent(self, mean, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * std

    def transform_latent(self, latent):
        transformed, log_qz = self.normalizing_flow(latent)
        return transformed, log_qz

    def call(self, x, training=False, apply_flow=True):
        
        mean, logvar = self.decode_allsteps(x)     

        z = self.sample_latent(mean, logvar)  # z ~ N(mean, std)

        if apply_flow:
            z_transformed, log_qz = self.transform_latent(z)
        else:
            z_transformed, log_qz = z, None

        return mean, logvar, z_transformed, log_qz

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)