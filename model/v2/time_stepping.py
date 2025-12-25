import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from model.normalizing_flow import PlanarFlow, InvertibleFlow, NormalizingFlow


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class EncoderBlock(layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(4 * dim, activation='relu'),
            layers.Dense(dim),
        ])
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.layer_norm1(inputs)
        attn_output = self.attention(x, x)
        x = self.dropout1(attn_output) + inputs
        ff_output = self.feed_forward(self.layer_norm2(x))
        return self.dropout2(ff_output) + x
    
    def compute_output_shape(self, input_shape):
        return input_shape  
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config


class DecoderBlock(layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(4 * dim, activation='relu'),
            layers.Dense(dim),
        ])
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_output):
        x = self.layer_norm1(inputs)
        attn_output = self.self_attention(x, x)
        x = self.dropout1(attn_output) + inputs
        x = self.layer_norm2(x)
        attn_output = self.cross_attention(x, encoder_output)
        x = self.dropout2(attn_output) + x
        ff_output = self.feed_forward(self.layer_norm3(x))
        return self.dropout3(ff_output) + x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    
class VariationalTimeSteppingTransformerV2(tf.keras.Model):
    def __init__(self, latent_dim, num_blocks, num_heads, dim, dropout_rate, context_window, max_prediction_steps=12, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.context_window = context_window
        self.max_prediction_steps = max_prediction_steps

        # Encoder
        self.encoder_embedding = tf.keras.layers.Dense(dim)
        self.encoder_positional_encoding = self.positional_encoding(context_window, dim)
        self.encoder_blocks = [EncoderBlock(dim, num_heads, dropout_rate) for _ in range(num_blocks)]

        # Decoder
        self.decoder_embedding = tf.keras.layers.Dense(dim)
        self.decoder_positional_encoding = self.positional_encoding(max_prediction_steps, dim)
        self.decoder_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_blocks = [DecoderBlock(dim, num_heads, dropout_rate) for _ in range(num_blocks)]

        # Output
        self.final_mean_layer = tf.keras.layers.Dense(latent_dim)
        self.final_logvar_layer = tf.keras.layers.Dense(latent_dim)

    def positional_encoding(self, max_len, d_model):
        angle_rads = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis] / tf.pow(
            tf.cast(10000, dtype=tf.float32), tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def encode_context(self, inputs, training=False):
        x = self.encoder_embedding(inputs)
        x += self.encoder_positional_encoding
        for block in self.encoder_blocks:
            x = block(x, training=training)
        return x

    def decode_allsteps(self, decoder_input_sequence, encoder_output, training=False):
        x = self.decoder_embedding(decoder_input_sequence)
        x += self.decoder_positional_encoding[:, :self.max_prediction_steps, :]
        x = self.decoder_dropout(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, encoder_output, training=training)

        mean = self.final_mean_layer(x)
        logvar = self.soft_clip_tanh(self.final_logvar_layer(x), -6, 3)
        return mean, logvar

    def decode_onestep(self, latent, context, step, training=False, return_state=False):
        x = self.decoder_embedding(latent)
        x += tf.slice(self.decoder_positional_encoding, [0, step, 0], [1, 1, self.dim])
        x = self.decoder_dropout(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, context, training=training)

        mean = self.final_mean_layer(x)
        logvar = self.soft_clip_tanh(self.final_logvar_layer(x), -6, 3)
        return (mean, logvar, x) if return_state else (mean, logvar)

    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))

    def call(self, inputs, output_steps=None, training=False):
        output_steps = output_steps or self.max_prediction_steps
        encoder_output = self.encode_context(inputs, training=training)
        start_token = inputs[:, -1:]
        decoder_inputs = tf.repeat(start_token, repeats=output_steps, axis=1)
        return self.decode_allsteps(decoder_inputs, encoder_output, training=training)

    def compute_output_shape(self, input_shape):
        return [
            input_shape[:1] + (self.max_prediction_steps, self.latent_dim),
            input_shape[:1] + (self.max_prediction_steps, self.latent_dim)
        ]

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dim': self.dim,
            'dropout_rate': self.dropout_rate,
            'context_window': self.context_window,
            'max_prediction_steps': self.max_prediction_steps
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    


class VariationalTimeSteppingTransformerNF2(VariationalTimeSteppingTransformerV2):
    def __init__(self, num_flows=3, **kwargs):
        super().__init__(**kwargs)
        self.num_flows = num_flows
        self.flow_layers = self._build_flows()
        self.normalizing_flow = NormalizingFlow(self.flow_layers)

    def _build_flows(self):
        # PlanarFlow & InvertibleFlow
        flows = []
        for i in range(self.num_flows):
            if i % 2 == 0:
                flows.append(PlanarFlow(self.latent_dim))
            else:
                flows.append(InvertibleFlow(self.latent_dim))
        return flows

    def sample_latent(self, mean, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * std

    def transform_latent(self, latent):
        return self.normalizing_flow(latent)

    def call(self, inputs, output_steps=None, training=False):
        output_steps = output_steps or self.max_prediction_steps

        encoder_output = self.encode_context(inputs, training=training)
        start_token = inputs[:, -1:]
        decoder_inputs = tf.repeat(start_token, repeats=output_steps, axis=1)

        latent_means, latent_logvars = self.decode_allsteps(decoder_inputs, encoder_output, training=training)
        latent_samples = self.sample_latent(latent_means, latent_logvars)
        latent_transformed, log_qz_vals = self.transform_latent(latent_samples)

        return latent_means, latent_logvars, latent_transformed, log_qz_vals

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        return [
            input_shape[:1] + [self.max_prediction_steps, self.latent_dim],
            input_shape[:1] + [self.max_prediction_steps, self.latent_dim],
            input_shape[:1] + [self.max_prediction_steps, self.latent_dim],
            input_shape[:1] + [self.max_prediction_steps],
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_flows': self.num_flows,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
