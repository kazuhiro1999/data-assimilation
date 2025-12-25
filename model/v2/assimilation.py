import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.wae import GCNLayer
from model.normalizing_flow import PlanarFlow, InvertibleFlow, NormalizingFlow
from utils.metrics import MPJPEError, MPJPEErrorFrame


class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, dropout_rate=0.1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.dropout = layers.Dropout(dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(4 * dim, activation=activation),
            layers.Dropout(dropout_rate),
            layers.Dense(dim),
        ])
        self.ffn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, context, training=False):
        norm_q = self.layer_norm1(query)
        norm_ctx = self.layer_norm2(context)
        attn_out = self.attn(norm_q, norm_ctx, norm_ctx, training=training)
        out = query + self.dropout(attn_out, training=training)
        ffn_out = self.ffn(out, training=training)
        return out + self.ffn_dropout(ffn_out, training=training)

class MultiStepPredictionLayerObs(tf.keras.layers.Layer):
    def __init__(self, latent_dim=64, dim=256, num_heads=4, num_blocks=2,
                 dropout_rate=0.1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.obs_proj = layers.Dense(dim)
        self.ctx_proj = layers.Dense(dim)
        self.time_embedding = layers.Embedding(input_dim=1000, output_dim=dim)

        # 初期生成用
        self.init_cross = CrossAttentionBlock(dim, num_heads, dropout_rate, activation)

        # 補正用ブロック（交互に観測・過去系列を参照）
        self.cross_blocks = []
        for i in range(num_blocks):
            self.cross_blocks.append(CrossAttentionBlock(dim, num_heads, dropout_rate, activation))  # obs
            self.cross_blocks.append(CrossAttentionBlock(dim, num_heads, dropout_rate, activation))  # context

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mean_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)

    def add_time_embedding(self, x, offset=0):
        batch_size, seq_len, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        positions = tf.range(seq_len + offset)[tf.newaxis, offset:]
        pos_embed = self.time_embedding(positions)
        return x + pos_embed

    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid + half_range * tf.tanh((x - mid) / (half_range * scale))

    def call(self, context, latent_obs, training=False):
        # (B, T, D)
        obs = self.obs_proj(latent_obs)
        ctx = self.ctx_proj(context)

        obs = self.add_time_embedding(obs, offset=30)
        ctx = self.add_time_embedding(ctx)

        # 初期状態（上半身観測 × 過去系列）で全身初期予測
        x = self.init_cross(obs, ctx, training=training)

        # 交互に Cross-Attention 適用
        for i, block in enumerate(self.cross_blocks):
            if i % 2 == 0:
                # 偶数番目は観測参照（補完）
                x = block(x, obs, training=training)
            else:
                # 奇数番目は過去系列参照（時間整合性補正）
                x = block(x, ctx, training=training)

        x = self.final_norm(x)
        mean = self.mean_layer(x)
        logvar = self.soft_clip_tanh(self.logvar_layer(x), min_val=-6, max_val=3)
        return mean, logvar

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (
            tf.TensorShape([batch_size, None, self.latent_dim]),
            tf.TensorShape([batch_size, None, self.latent_dim])
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config

    
class ObservationEncoder(tf.keras.layers.Layer):
    def __init__(self, dims, latent_dim, dropout_rate=0.0, **kwargs):
        super(ObservationEncoder, self).__init__(**kwargs)
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
        config = super(ObservationEncoder, self).get_config()
        config.update({
            'dims': self.dims,
            'latent_dim': self.latent_dim,
            'dropout_rate': self.dropout_rate
        })
        return config
    

    
class DeepLatentSpaceAssimilationModelV4(keras.Model):
    def __init__(
        self, 
        wae, 
        latent_dim, 
        num_joints, 
        context_window, 
        prediction_steps, 
        num_flows,
        obs_encoder_dims=[512, 256],
        obs_dropout_rate=0.2,
        pred_dim=256,
        pred_num_heads=4,
        pred_num_blocks=1,
        pred_dropout_rate=0.2,
        nll_weight=0.1,
        dtw_weight=1.0,
        mse_weight=1.0,
        **kwargs
    ):
        super(DeepLatentSpaceAssimilationModelV4, self).__init__(**kwargs)
        
        self.wae = wae
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.context_window = context_window
        self.prediction_steps = prediction_steps
        self.num_flows = num_flows
        self.obs_encoder_dims = obs_encoder_dims
        self.obs_dropout_rate = obs_dropout_rate
        self.pred_dim = pred_dim
        self.pred_num_heads = pred_num_heads
        self.pred_num_blocks = pred_num_blocks
        self.pred_dropout_rate = pred_dropout_rate
        self.nll_weight = nll_weight
        self.dtw_weight = dtw_weight
        self.mse_weight = mse_weight        
        
        # Observation encoder
        self.observation_layer = ObservationEncoder(
            dims=obs_encoder_dims, 
            latent_dim=latent_dim, 
            dropout_rate=obs_dropout_rate
        )
        self.observation_model = layers.TimeDistributed(
            self.observation_layer, name="observation_model_td"
        )
        
        # Integrating layer
        self.prediction_layer = MultiStepPredictionLayerObs(
            latent_dim=latent_dim, 
            dim=pred_dim, 
            num_heads=pred_num_heads, 
            num_blocks=pred_num_blocks, 
            dropout_rate=pred_dropout_rate
        )
        
        #self.gate_layer = SelectionGatesAttention(latent_dim, dim=latent_dim*2, num_heads=4, dropout_rate=0.2)
        
        self.flow_layers = self._build_flows(num_flows, latent_dim)
        self.normalizing_flow = NormalizingFlow(self.flow_layers)
                
        wae.trainable = False

        # WAE Encoder
        encoder_input = keras.Input(shape=(self.num_joints, 3))
        encoder_output = self.wae.encode(encoder_input)
        self.wae_encoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=encoder_input, outputs=encoder_output, name="WAE Encoder"),
            name="wae_encoder_td"
        )    
        
        # WAE Decoder
        final_input = keras.Input(shape=(self.latent_dim,))
        final_output = self.wae.decode(final_input)
        self.wae_decoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=final_input, outputs=final_output, name="WAE Decoder"),
            name="wae_decoder_td"
        )               
        
        # Loss weight
        self.latent_nll_weight = nll_weight  # 潜在空間でのNLL
        self.dtw_weight = dtw_weight
        self.mse_weight = mse_weight
        
        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.latent_nll_tracker = keras.metrics.Mean(name="latent_nll_loss")
        self.dtw_loss_tracker = keras.metrics.Mean(name="dtw_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        
        self.metrics_list = [
            MPJPEError(name="mpjpe_error"),
            MPJPEErrorFrame(3, name="mpjpe@100ms"),
        ]
        
    def _build_flows(self, num_flows, latent_dim):
        # 交互にPlanarFlowとInvertibleFlowを加える
        flows = []
        for i in range(num_flows):
            if i % 2 == 0:
                flows.append(PlanarFlow(latent_dim))
            else:
                flows.append(InvertibleFlow(latent_dim))
        return flows
    
    def sample_latent(self, mean, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * std

    def transform_latent(self, latent):
        return self.normalizing_flow(latent)
    
    def soft_clip_tanh(self, x, min_val, max_val, scale=1.0):
        mid_val = (max_val + min_val) / 2
        half_range = (max_val - min_val) / 2
        return mid_val + half_range * tf.tanh((x - mid_val) / (half_range * scale))  # Center around zero, apply tanh, then scale and shift back

    def call(self, inputs, training=False, return_state=False):
        past_poses, observations = inputs
        batch_size = tf.shape(observations)[0]

        # --- Encode past pose sequence to latent space ---
        past_states = self.wae_encoder(past_poses)  # (batch_size, context_length, latent_dim)
        
        # --- Encode observation data ---
        latent_obs = self.observation_model(observations)  # => (batch_size, prediction_steps, latent_dim)
        
        # --- Prediction All Steps ---
        latent_means, latent_logvars = self.prediction_layer(past_states, latent_obs, training=training)  # (batch_size, prediction_steps, latent_dim)           

        # --- サンプリング ---
        if training:
            latent_samples = self.sample_latent(latent_means, latent_logvars)  
        else:
            latent_samples = latent_means    

        # --- Normalizing Flow 適用 ---
        latent_transformed, log_qz_vals = self.transform_latent(latent_samples)  # (batch_size, prediction_steps, latent_dim), (B, T)
        
        decoded_poses = self.wae_decoder(latent_transformed)

        if not return_state:
            return decoded_poses
        else:
            return decoded_poses, (latent_means, latent_logvars, latent_transformed, log_qz_vals)

    def compute_latent_nll_loss(self, latent_true, latent_pred_mean, latent_pred_log_var, log_qz):
        """潜在空間での負の対数尤度損失"""
        squared_error = tf.square(latent_true - latent_pred_mean)
        variance = tf.exp(latent_pred_log_var) + 1e-6

        nll = 0.5 * (tf.math.log(2 * np.pi * variance) + squared_error / variance)
        return tf.reduce_mean(nll)
    
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
            "latent_dim": self.latent_dim,
            "context_window": self.context_window,
            "prediction_steps": self.prediction_steps,
            "num_joints": self.num_joints,            
            "num_flows": self.num_flows,
            "obs_encoder_dims": self.obs_encoder_dims,
            "obs_dropout_rate": self.obs_dropout_rate,
            "pred_dim": self.pred_dim,
            "pred_num_heads": self.pred_num_heads,
            "pred_num_blocks": self.pred_num_blocks,
            "pred_dropout_rate": self.pred_dropout_rate,
            "nll_weight": self.nll_weight,
            "dtw_weight": self.dtw_weight,
            "mse_weight": self.mse_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """ モデルの設定をロードするためのメソッド """
        wae = keras.utils.deserialize_keras_object(config.pop("wae"))  # サブモデルをデシリアライズ

        return cls(
            wae=wae,
            **config 
        )