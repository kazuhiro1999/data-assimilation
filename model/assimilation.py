import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model.wae import GCNLayer
from utils.metrics import MPJPEError, MPJPEErrorFrame


class SelectionGatesAttention(tf.keras.layers.Layer):
    def __init__(self, latent_dim, dim=256, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim  
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.obs_projection = layers.Dense(self.dim)

        # Cross-Attention for fusion
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim)
        self.dropout1 = layers.Dropout(self.dropout_rate)

        # Feed-Forward Network
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(4 * self.dim, activation='relu'),
            layers.Dense(self.dim),
        ])
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.final_layer = layers.Dense(self.latent_dim, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=False):
        """ 
        - latent_pred: 予測状態 (batch_size, dim)
        - latent_obs: 観測情報の潜在表現 (batch_size, dim)
        """
        latent_pred, latent_obs = inputs
        
        # Expand dim        
        latent_pred = tf.expand_dims(latent_pred, axis=1) 
        latent_obs = tf.expand_dims(latent_obs, axis=1) 
        
        # latent_pred を dim に統一
        latent_obs = self.obs_projection(latent_obs)  # (batch_size, 1, dim)

        # Cross-Attention: 予測状態と観測情報を統合
        x = self.layer_norm1(latent_pred)
        attn_output = self.cross_attention(x, latent_obs, latent_obs)
        x = self.dropout1(attn_output, training=training) + x

        # Feed-Forward Network
        ff_output = self.feed_forward(self.layer_norm2(x))
        ff_output = self.dropout2(ff_output, training=training) + x
        
        gates = self.final_layer(ff_output)
        return tf.squeeze(gates, axis=1)
    
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0][:-1]) + (self.latent_dim,)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config
    

class FusionLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.dim = dim  # 次元は統一する必要がある
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):

        self.obs_projection = layers.Dense(self.dim)

        # Self-Attention for latent_pred
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim)
        self.dropout1 = layers.Dropout(self.dropout_rate)

        # Cross-Attention for fusion
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attention = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim)
        self.dropout2 = layers.Dropout(self.dropout_rate)

        # Feed-Forward Network
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(4 * self.dim, activation='relu'),
            layers.Dense(self.dim),
        ])
        self.dropout3 = layers.Dropout(self.dropout_rate)
        
        self.mean_layer = layers.Dense(self.dim//2)
        self.logvar_layer = layers.Dense(self.dim//2)

    def call(self, latent_pred, latent_obs, training=False):
        """ 
        - latent_pred: 予測状態 (batch_size, dim)
        - latent_obs: 観測情報の潜在表現 (batch_size, dim)
        """        
        # Expand dim        
        latent_pred = tf.expand_dims(latent_pred, axis=1) 
        latent_obs = tf.expand_dims(latent_obs, axis=1) 
        
        # latent_pred を dim に統一
        latent_obs = self.obs_projection(latent_obs)  # (batch_size, 1, dim)

        # Self-Attention: 予測状態の内部関係を学習
        x = self.layer_norm1(latent_pred)
        attn_output = self.self_attention(x, x)
        x = self.dropout1(attn_output, training=training) + latent_pred

        # Cross-Attention: 予測状態と観測情報を統合
        x = self.layer_norm2(x)
        attn_output = self.cross_attention(x, latent_obs)
        x = self.dropout2(attn_output, training=training) + x

        # Feed-Forward Network
        ff_output = self.feed_forward(self.layer_norm3(x))
        ff_output = self.dropout3(ff_output, training=training) + x
        
        mean = self.mean_layer(ff_output)
        logvar = self.logvar_layer(ff_output)
        logvar = self.soft_clip_tanh(logvar, min_val=-6, max_val=3)
        
        return tf.squeeze(mean, axis=1), tf.squeeze(logvar, axis=1)
    
    @tf.function
    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))
    
    def compute_output_shape(self, input_shape):
        latent_dim = self.dim // 2
        return (input_shape[0][:-1] + (latent_dim,)), (input_shape[0][:-1] + (latent_dim,))
    
    def get_config(self):
        config = super(FusionLayer, self).get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return config

    
class ObservationModel(tf.keras.layers.Layer):
    def __init__(self, dims, latent_dim, dropout_rate=0.0):
        super().__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        self.gcn_layers = [GCNLayer(units=dim, dropout_rate=self.dropout_rate) for dim in self.dims]
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.final_layer = layers.Dense(
            self.latent_dim, 
            kernel_regularizer=keras.regularizers.L2(0.01), 
            bias_regularizer=keras.regularizers.L2(0.01), 
            activity_regularizer=keras.regularizers.L2(0.01)
        )
        
    def build(self, input_shape):
        super().build(input_shape)        
        
    def call(self, inputs, training=False):
        x = inputs
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x)        
        x = self.global_pooling(x)
        output = self.final_layer(x)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (self.latent_dim,)
    
    def get_config(self):
        config = super(ObservationModel, self).get_config()
        config.update({
            'dims': self.dims,
            'latent_dim': self.latent_dim,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    
class ObsLikelihoodEstimator(tf.keras.layers.Layer):
    def __init__(self, dims, latent_dim, dropout_rate=0.0):
        super().__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.fc_layers = [layers.Dense(dim, activation='relu') for dim in self.dims]
        self.final_layer = layers.Dense(self.latent_dim, activation='linear')
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = inputs
        for fc_layer in self.fc_layers:
            x = fc_layer(x)        
        output = self.final_layer(x)
        return output
    
    def get_config(self):
        config = super(ObservationModel, self).get_config()
        config.update({
            'dims': self.dims,
            'latent_dim': self.latent_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

    
class DeepLatentSpaceAssimilationModel(keras.Model):
    def __init__(self, wae, time_stepping_model, latent_dim, num_joints, context_window, prediction_steps, **kwargs):
        super().__init__(**kwargs)
        self.wae = wae
        self.time_stepping_model = time_stepping_model
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.context_window = context_window
        self.prediction_steps = prediction_steps
        
        self.observation_model = ObservationModel(dims=[512, 256], latent_dim=64, dropout_rate=0.2)
        self.fusion_layer = FusionLayer(dim=latent_dim * 2, num_heads=4, dropout_rate=0.2)
        self.gate_layer = SelectionGatesAttention(latent_dim, dim=latent_dim*2, num_heads=4, dropout_rate=0.2)
                
        wae.trainable = False
        time_stepping_model.encoder.trainable = False

        self.wae_encoder = None
        self.wae_decoder = None
        
        self.initial_logvar = tf.Variable(
            initial_value=tf.fill([1, 1, latent_dim], -5.0), trainable=True, dtype=tf.float32
        )
        
        self.latent_nll_weight = 0.001  # NLL in latent space
        self.dtw_weight = 0.1
        self.mse_weight = 100.0
        
        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.latent_nll_tracker = keras.metrics.Mean(name="latent_nll_loss")
        self.dtw_loss_tracker = keras.metrics.Mean(name="dtw_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        
        self.metrics_list = [
            MPJPEError(name="mpjpe_error"),
            MPJPEErrorFrame(3, name="mpjpe@100ms"),
        ]
        
    def build(self, input_shape):
        super().build(input_shape)    
        
        # WAE Encoder
        encoder_input = keras.Input(shape=(self.num_joints, 3))
        encoder_output = self.wae.encode(encoder_input)
        self.wae_encoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=encoder_input, outputs=encoder_output, name="WAE Encoder")
        )        
        self.wae_encoder.build(input_shape=input_shape)

        # WAE Decoder
        final_input = keras.Input(shape=(self.latent_dim,))
        final_output = self.wae.decode(final_input)
        self.wae_decoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=final_input, outputs=final_output, name="WAE Decoder")
        )        
        self.wae_decoder.build(input_shape=input_shape[:1] + (self.prediction_steps, self.latent_dim))
        
        self.observation_model.build(input_shape=(None, None, 3))
        self.fusion_layer.build(input_shape=[(None, 1, self.latent_dim), (None, 1, self.latent_dim)])
        self.gate_layer.build(input_shape=[(None, 1, self.latent_dim), (None, 1, self.latent_dim)])

    def call(self, inputs, training=False, return_state=False):
        past_poses, observations = inputs
        batch_size = tf.shape(observations)[0]

        # Encode input data to latent space
        past_states = self.wae_encoder(past_poses)  # (batch_size, input_seq_len, latent_dim)
        
        # Prepare time stepping inputs
        transformer_context = self.time_stepping_model.encode_context(past_states)  # => (batch_size, context_window, dim)
        latent_mean = past_states[:, -1:]
        latent_logvar = tf.tile(self.initial_logvar, [batch_size, 1, 1])
        latent_transformed = latent_mean

        means = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        logvars = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        samples = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        log_qz_vals = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        
        for step in range(self.prediction_steps):       
            
            # 1.Apply Time Stepping             
            latent_pred_mean, latent_pred_logvar = self.time_stepping_model.decode_onestep(
                latent_transformed, 
                transformer_context, 
                step,
                training=training,
                return_state=False
            )        
            
            # 2.Embed Observations
            latent_obs = self.observation_model(observations[:, step])  # (batch_size, latent_dim)
            latent_obs = tf.expand_dims(latent_obs, axis=1)
            
            # 3.Distribution Proposal: 予測と観測に基づいて分布を提案（Cross-Attention）
            latent_pred = tf.concat([latent_pred_mean, latent_pred_logvar], axis=-1)  # (batch_size, 1, latent_dim * 2)
            updated_mean, updated_logvar = self.fusion_layer(latent_pred, latent_obs, training=training)  # => (batch_size, 1, latent_dim)
            
            # 4.現在の状態を選択的に更新
            current_state = tf.concat([latent_mean, latent_logvar], axis=-1)  # (batch_size, 1, latent_dim * 2)
            updated_state = tf.concat([updated_mean, updated_logvar], axis=-1)  # (batch_size, 1, latent_dim * 2)
            gates = self.gate_layer([current_state, updated_state], training=training)  # => (batch_size, 1, latent_dim)
            
            latent_mean = (1 - gates) * latent_mean + gates * updated_mean
            latent_logvar = tf.math.reduce_logsumexp(tf.stack([
                tf.math.log(1 - gates**2) + latent_logvar, 
                tf.math.log(gates**2) + updated_logvar
            ], axis=-1), axis=-1)
            
            # 5.Sampling (Reparameterization trick)
            if training:
                latent_sampled = self.time_stepping_model.sample_latent(latent_mean, latent_logvar)  
            else:
                latent_sampled = latent_mean         
            latent_transformed, log_qz_transformed = self.time_stepping_model.transform_latent(latent_sampled)  # Apply Normalizing Flow
            
            means = means.write(step, latent_mean)
            logvars = logvars.write(step, latent_logvar)
            samples = samples.write(step, latent_transformed)
            log_qz_vals = log_qz_vals.write(step, log_qz_transformed)

        # Decode future states        
        means = tf.transpose(means.stack(), [1, 0, 2, 3])  # (batch_size, output_steps, latent_dim)
        logvars = tf.transpose(logvars.stack(), [1, 0, 2, 3])
        samples = tf.transpose(samples.stack(), [1, 0, 2, 3])
        log_qz_vals = tf.transpose(log_qz_vals.stack(), [1, 0, 2])
        
        means = tf.squeeze(means, axis=2)
        logvars = tf.squeeze(logvars, axis=2)
        samples = tf.squeeze(samples, axis=2)
        log_qz_vals = tf.squeeze(log_qz_vals, axis=2)
        
        decoded_poses = self.wae_decoder(samples)

        if not return_state:
            return decoded_poses
        else:
            return decoded_poses, (means, logvars, samples, log_qz_vals)
    
    @tf.function
    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))  # Center around zero, apply tanh, then scale and shift back
    
    @tf.function
    def compute_latent_nll_loss(self, latent_true, latent_pred_mean, latent_pred_log_var, log_qz):
        """Negative Log-Liklihood in Latent Space"""
        squared_error = tf.square(latent_true - latent_pred_mean)
        variance = tf.exp(latent_pred_log_var) + 1e-6

        nll = 0.5 * (tf.math.log(2 * np.pi * variance) + squared_error / variance)
        return tf.reduce_mean(nll)
    
    @tf.function
    def compute_dtw_loss(self, y_true, y_pred, max_batch_size=64):
        """Dynamic Time Warping"""       
        batch_size = tf.shape(y_true)[0]
        seq_len = tf.shape(y_true)[1]  # Sequence length
        
        y_true = tf.reshape(y_true, [batch_size, seq_len, -1])
        y_pred = tf.reshape(y_pred, [batch_size, seq_len, -1])

        # Initialize matrix
        dtw_matrix = tf.fill(
            [max_batch_size, seq_len + 1, seq_len + 1],
            tf.constant(tf.float32.max, dtype=tf.float32)
        )

        dtw_matrix = tf.tensor_scatter_nd_update(
            dtw_matrix,
            tf.constant([[i, 0, 0] for i in range(max_batch_size)]),
            tf.zeros(max_batch_size, dtype=tf.float32)
        )

        # function: Update DTW Matrix
        def dtw_step(i, dtw_matrix):
            # Calculate cost
            cost = tf.reduce_sum(tf.square(y_pred[:, i - 1] - y_true[:, i - 1]), axis=-1)  # [batch_size]

            indices = tf.concat(
                [
                    tf.expand_dims(tf.range(batch_size), axis=1),  
                    tf.fill([batch_size, 1], i),                  
                    tf.fill([batch_size, 1], i)                   
                ],
                axis=1
            )
            updates = cost + tf.minimum(
                dtw_matrix[:batch_size, i - 1, i - 1],
                tf.minimum(dtw_matrix[:batch_size, i - 1, i], dtw_matrix[:batch_size, i, i - 1])
            )
            dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, indices, updates)
            return dtw_matrix

        # Calculate DTW Matrix
        for i in tf.range(1, seq_len + 1):
            dtw_matrix = dtw_step(i, dtw_matrix)

        dtw_distance = dtw_matrix[:batch_size, -1, -1]
        return dtw_distance
    
    @property
    def metrics(self):
        return [self.train_loss_tracker, self.val_loss_tracker, self.latent_nll_tracker, self.dtw_loss_tracker, self.mse_loss_tracker] + self.metrics_list

    @tf.function
    def train_step(self, data):
        inputs, y_true = data
        past_poses, observations = inputs
        latent_true = self.wae_encoder(y_true)  # Encode ground truth to latent space
        
        with tf.GradientTape() as tape:
            y_pred, (latent_mean, latent_log_var, latent_sampled, log_qz) = self((past_poses, observations), training=True, return_state=True)
            
            latent_nll_loss = self.compute_latent_nll_loss(latent_true, latent_mean, latent_log_var, log_qz)
            dtw_loss = self.compute_dtw_loss(y_true, y_pred) if self.dtw_weight > 0 else 0.0
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true[:,:3], y_pred[:,:3]))    
            total_loss = (
                self.latent_nll_weight * latent_nll_loss +
                self.mse_weight * mse_loss +
                self.dtw_weight * dtw_loss
            )
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.train_loss_tracker.update_state(total_loss)
        self.latent_nll_tracker.update_state(latent_nll_loss)
        self.dtw_loss_tracker.update_state(dtw_loss)
        self.mse_loss_tracker.update_state(mse_loss)

        for metric in self.metrics_list:
            metric.update_state(y_true, y_pred)
            
        return {
            "loss": self.train_loss_tracker.result(), 
            "latent_nll": self.latent_nll_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics_list}
        }

    @tf.function
    def test_step(self, data):
        inputs, y_true = data
        past_poses, observations = inputs
        latent_true = self.wae_encoder(y_true)  # Encode ground truth to latent space
        
        y_pred, (latent_mean, latent_log_var, latent_sampled, log_qz) = self((past_poses, observations), training=False, return_state=True)
        
        latent_nll_loss = self.compute_latent_nll_loss(latent_true, latent_mean, latent_log_var, log_qz)
        dtw_loss = self.compute_dtw_loss(y_true, y_pred) if self.dtw_weight > 0 else 0.0
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true[:,:3], y_pred[:,:3]))        
        total_loss = (
            self.latent_nll_weight * latent_nll_loss +
            self.mse_weight * mse_loss +
            self.dtw_weight * dtw_loss 
        )
        
        self.val_loss_tracker.update_state(total_loss)
        self.latent_nll_tracker.update_state(latent_nll_loss)
        self.dtw_loss_tracker.update_state(dtw_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        
        for metric in self.metrics_list:
            metric.update_state(y_true, y_pred)
            
        return {
            "loss": self.val_loss_tracker.result(), 
            "latent_nll": self.latent_nll_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics_list}
        }
    
    def get_config(self):
        config = super().get_config() 
        config.update({
            "wae": keras.utils.serialize_keras_object(self.wae),  
            "time_stepping_model": keras.utils.serialize_keras_object(self.time_stepping_model),
            "latent_dim": self.latent_dim,
            "context_window": self.context_window,
            "prediction_steps": self.prediction_steps,
            "num_joints": self.num_joints
        })
        return config

    @classmethod
    def from_config(cls, config):
        """ モデルの設定をロードするためのメソッド """
        wae = keras.utils.deserialize_keras_object(config.pop("wae"))  # Deserialize Sub Model
        time_stepping_model = keras.utils.deserialize_keras_object(config.pop("time_stepping_model"))

        return cls(
            wae=wae,
            time_stepping_model=time_stepping_model,
            **config 
        )