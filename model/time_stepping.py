import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from model.normalizing_flow import PlanarFlow, InvertibleFlow, NormalizingFlow


class EncoderBlock(layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
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


class DecoderBlock(layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
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

    
class VariationalTimeSteppingTransformer(Model):
    def __init__(self, latent_dim, num_blocks, num_heads, dim, dropout_rate, context_window, max_prediction_steps=12, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.context_window = context_window
        self.max_prediction_steps = max_prediction_steps

        # Encoder components
        self.encoder_input = layers.Input(shape=(context_window, latent_dim), name="encoder_input")
        self.encoder_embedding = layers.Dense(dim)
        self.encoder_positional_encoding = self.positional_encoding(context_window, dim)
        self.encoder_blocks = [EncoderBlock(dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.encoder = None

        # Decoder components
        self.decoder_embedding = layers.Dense(dim)
        self.decoder_positional_encoding = self.positional_encoding(self.max_prediction_steps, dim)
        self.decoder_dropout = layers.Dropout(dropout_rate)
        self.decoder_blocks = [DecoderBlock(dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.decoder = None

        self.final_mean_layer = layers.Dense(latent_dim, name="final_mean_layer")
        self.final_logvar_layer = layers.Dense(latent_dim, name="final_logvar_layer")
        
    def build(self, input_shape):
        # Encoder Model
        encoder_output = self.encoder_input
        encoder_output = self.encoder_embedding(encoder_output)
        encoder_output += self.encoder_positional_encoding
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)
        self.encoder = Model(self.encoder_input, encoder_output, name="TransformerEncoder")

        # Decoder Model 
        decoder_input = layers.Input(shape=(1, self.dim), name="decoder_input")
        encoder_output = layers.Input(shape=(self.context_window, self.dim), name="encoder_output")

        x = decoder_input
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
        x = layers.Dense(self.dim)(x)
        self.decoder = Model([decoder_input, encoder_output], x, name="TransformerDecoder")
        
        # Build other layers with correct shapes
        self.decoder_embedding.build(input_shape=(None, 1, self.latent_dim))
        self.decoder_dropout.build(input_shape=(None, 1, self.dim))
        self.final_mean_layer.build(input_shape=(None, 1, self.dim))
        self.final_logvar_layer.build(input_shape=(None, 1, self.dim))
        
        super().build(input_shape)

    def positional_encoding(self, max_len, d_model):
        """Standard Transformer positional encoding"""
        angle_rads = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis] / tf.pow(
            tf.cast(10000, dtype=tf.float32), tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]
    
    @tf.function
    def encode_context(self, inputs):
        return self.encoder(inputs)
    
    def decode_onestep(self, latent, context, step, training=False, return_state=False):
        # Embed and add positional encoding            
        decoder_input = self.decoder_embedding(latent)
        decoder_input += tf.slice(self.decoder_positional_encoding, [0, step, 0], [1, 1, self.dim])
        decoder_input = self.decoder_dropout(decoder_input, training=training)
        
        decoder_output = self.decoder([decoder_input, context], training=training)
        latent_pred_mean = self.final_mean_layer(decoder_output)
        latent_pred_logvar = self.final_logvar_layer(decoder_output)
        latent_pred_logvar = self.soft_clip_tanh(latent_pred_logvar, min_val=-6, max_val=3)
        
        if return_state:
            return latent_pred_mean, latent_pred_logvar, decoder_output
        
        return latent_pred_mean, latent_pred_logvar
    
    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2

        # Center around zero, apply tanh, then scale and shift back
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))

    @tf.function
    def call(self, inputs, output_steps=1, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Encoder processing
        encoder_output = self.encode_context(inputs)
        
        # Initialize decoder input
        decoder_input = inputs[:, -1:]
        
        means = tf.TensorArray(dtype=tf.float32, size=output_steps)
        logvars = tf.TensorArray(dtype=tf.float32, size=output_steps)
        
        for step in range(output_steps):
            # Generate prediction
            latent_pred_mean, latent_pred_logvar, decoder_output = self.decode_onestep(
                decoder_input, 
                encoder_output, 
                step,
                training=training,
                return_state=True
            )
            
            means = means.write(step, latent_pred_mean)
            logvars = logvars.write(step, latent_pred_logvar)
            
            decoder_input = latent_pred_mean
            
        means = means.stack()
        logvars = logvars.stack()
        
        means = tf.transpose(means, [1, 0, 2, 3])  # (batch_size, output_steps, latent_dim)
        logvars = tf.transpose(logvars, [1, 0, 2, 3])
        
        means = tf.squeeze(means, axis=2)
        logvars = tf.squeeze(logvars, axis=2)
        
        return means, logvars  # (batch_size, output_steps, latent_dim)
    
        
    def compute_output_shape(self, input_shape):
        return [
            input_shape[:1] + [None, self.latent_dim],
            input_shape[:1] + [None, self.latent_dim]
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dim': self.dim,
            'context_window': self.context_window,
            'max_prediction_steps': self.max_prediction_steps,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    
class VariationalTimeSteppingTransformerNF(VariationalTimeSteppingTransformer):
    def __init__(self, num_flows=1, **kwargs):
        super().__init__(**kwargs)
        self.num_flows = num_flows
        self.normalizing_flow = NormalizingFlow([
            PlanarFlow(self.latent_dim),
            InvertibleFlow(self.latent_dim),
            PlanarFlow(self.latent_dim),
        ])
        
    def build(self, input_shape):
        super().build(input_shape)
        self.normalizing_flow.build(input_shape)
        
    def sample_latent(self, mean, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mean))
        z_sampled = mean + eps * std
        return z_sampled

    def transform_latent(self, latent):
        latent_transformed, log_qz_transformed = self.normalizing_flow(latent)
        return latent_transformed, log_qz_transformed

    def call(self, inputs, output_steps=1, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Encoder processing
        encoder_output = self.encode_context(inputs)
        
        # Initialize decoder input
        decoder_input = inputs[:, -1:]
        
        means = tf.TensorArray(dtype=tf.float32, size=output_steps)
        logvars = tf.TensorArray(dtype=tf.float32, size=output_steps)
        transformed_samples = tf.TensorArray(dtype=tf.float32, size=output_steps)
        log_qz_vals = tf.TensorArray(dtype=tf.float32, size=output_steps)
        
        for step in range(output_steps):
            latent_mean, latent_logvar = self.decode_onestep(
                decoder_input, 
                encoder_output, 
                step,
                training=training,
                return_state=False
            )
            
            # Sampling
            latent_sampled = self.sample_latent(latent_mean, latent_logvar)
            
            # Apply Normalizing Flow
            latent_transformed, log_qz_transformed = self.transform_latent(latent_sampled)   
            
            means = means.write(step, latent_mean)
            logvars = logvars.write(step, latent_logvar)
            transformed_samples = transformed_samples.write(step, latent_transformed)
            log_qz_vals = log_qz_vals.write(step, log_qz_transformed)
            
            decoder_input = latent_transformed
            
        means = tf.transpose(means.stack(), [1, 0, 2, 3])  # (batch_size, output_steps, latent_dim)
        logvars = tf.transpose(logvars.stack(), [1, 0, 2, 3])
        transformed_samples = tf.transpose(transformed_samples.stack(), [1, 0, 2, 3])
        log_qz_vals = tf.transpose(log_qz_vals.stack(), [1, 0, 2])
        
        means = tf.squeeze(means, axis=2)
        logvars = tf.squeeze(logvars, axis=2)
        transformed_samples = tf.squeeze(transformed_samples, axis=2)
        log_qz_vals = tf.squeeze(log_qz_vals, axis=2)
        
        return means, logvars, transformed_samples, log_qz_vals
    
        
    def compute_output_shape(self, input_shape):
        return [
            input_shape[:1] + [None, self.latent_dim],
            input_shape[:1] + [None, self.latent_dim],
            input_shape[:1] + [None, self.latent_dim],
            input_shape[:1] + [None,],
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_flows': self.num_flows,
        })
        return config