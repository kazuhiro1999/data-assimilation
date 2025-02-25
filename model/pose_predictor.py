import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from utils.metrics import MPJPEError, MPJPEErrorFrame

class PosePredictionModel(tf.keras.Model):
    def __init__(self, wae, transformer, latent_dim, num_joints, context_window, prediction_steps, latent_nll_weight=None, dtw_weight=None, mse_weight=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.context_window = context_window
        self.prediction_steps = prediction_steps
        self.wae = wae  # WAEエンコーダ・デコーダ
        self.time_stepping_transformer = transformer
        
        wae.trainable = False
        
        # WAE Encoder
        encoder_input = keras.Input(shape=(num_joints, 3))
        encoder_output = self.wae.encode(encoder_input)
        self.wae_encoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=encoder_input, outputs=encoder_output, name="WAE Encoder")
        )

        # WAE Decoder
        decoder_input = keras.Input(shape=(latent_dim,))
        decoder_output = self.wae.decode(decoder_input)
        self.wae_decoder = tf.keras.layers.TimeDistributed(
            keras.Model(inputs=decoder_input, outputs=decoder_output, name="WAE Decoder")
        )
        
        self.latent_nll_weight = latent_nll_weight  # 潜在空間でのNLL
        self.dtw_weight = dtw_weight
        self.mse_weight = mse_weight
        
        # Metrics and Loss Trackers
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.latent_nll_tracker = keras.metrics.Mean(name="latent_nll_loss")
        self.dtw_loss_tracker = keras.metrics.Mean(name="dtw_loss")
        self.mse_loss_tracker = keras.metrics.Mean(name="mse_loss")
        
        self.metrics_list = [
            MPJPEError(name="mpjpe_error"),
            MPJPEErrorFrame(3, name="mpjpe@100ms"),
        ]

    def call(self, inputs, return_state=False, training=False):

        # Encode input data to latent space
        latent_past = self.wae_encoder(inputs)  # (batch_size, input_seq_len, latent_dim)
        
        # Prepare time stepping inputs
        transformer_context = self.time_stepping_transformer.encode_context(latent_past)  # => (batch_size, context_window, dim)
        decoder_input = latent_past[:, -1:]
        
        means = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        logvars = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        samples = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        log_qz_vals = tf.TensorArray(dtype=tf.float32, size=self.prediction_steps)
        
        for step in range(self.prediction_steps):
             
            mean, logvar = self.time_stepping_transformer.decode_onestep(
                decoder_input, 
                transformer_context, 
                step,
                training=training,
                return_state=False
            )
            
            # Sampling
            latent_sampled = self.time_stepping_transformer.sample_latent(mean, logvar)
            
            # Apply Normalizing Flow
            latent_transformed, log_qz_transformed = self.time_stepping_transformer.transform_latent(latent_sampled)          
            
            means = means.write(step, mean)
            logvars = logvars.write(step, logvar)
            samples = samples.write(step, latent_transformed)
            log_qz_vals = log_qz_vals.write(step, log_qz_transformed)
            
            decoder_input = latent_transformed  # 次のステップの入力としてサンプルを使用
        
        means = tf.transpose(means.stack(), [1, 0, 2, 3])  # (batch_size, output_steps, latent_dim)
        logvars = tf.transpose(logvars.stack(), [1, 0, 2, 3])
        samples = tf.transpose(samples.stack(), [1, 0, 2, 3])
        log_qz_vals = tf.transpose(log_qz_vals.stack(), [1, 0, 2])
        
        means = tf.squeeze(means, axis=2)
        logvars = tf.squeeze(logvars, axis=2)
        samples = tf.squeeze(samples, axis=2)
        log_qz_vals = tf.squeeze(log_qz_vals, axis=2)
        
        recon_poses = self.wae_decoder(samples)

        # return_state に応じて出力を切り替え
        if return_state:
            return recon_poses, (means, logvars, samples, log_qz_vals)  # 分布全体も返す
        else:
            return recon_poses  # 状態の推定値のみ
        
    @tf.function
    def compute_latent_nll_loss(self, latent_true, latent_pred_mean, latent_pred_log_var, log_qz):
        """潜在空間での負の対数尤度損失"""
        squared_error = tf.square(latent_true - latent_pred_mean)
        variance = tf.exp(latent_pred_log_var) + 1e-6

        nll = 0.5 * (tf.math.log(2 * np.pi * variance) + squared_error / variance)
        return tf.reduce_mean(nll - tf.expand_dims(log_qz, axis=-1))
    
    @tf.function
    def compute_dtw_loss(self, y_true, y_pred, max_batch_size=64):
        """Dynamic Time Warping損失を計算"""
        # パラメータ設定
        seq_len = tf.shape(y_true)[1]  # シーケンス長を動的に取得
        feature_dim = tf.shape(y_true)[2]  # 特徴次元数

        # バッチサイズの動的取得
        batch_size = tf.shape(y_true)[0]

        # 入力のリシェイプ
        y_true = tf.reshape(y_true, [batch_size, seq_len, -1])
        y_pred = tf.reshape(y_pred, [batch_size, seq_len, -1])

        # DTW行列の初期化
        dtw_matrix = tf.fill(
            [max_batch_size, seq_len + 1, seq_len + 1],
            tf.constant(tf.float32.max, dtype=tf.float32)
        )
        # (0, 0)位置を初期化
        dtw_matrix = tf.tensor_scatter_nd_update(
            dtw_matrix,
            tf.constant([[i, 0, 0] for i in range(max_batch_size)]),
            tf.zeros(max_batch_size, dtype=tf.float32)
        )

        # ヘルパー関数: DTW行列の更新
        def dtw_step(i, dtw_matrix):
            # 現在のコストを計算
            cost = tf.reduce_sum(tf.square(y_pred[:, i - 1] - y_true[:, i - 1]), axis=-1)  # [batch_size]

            # 更新する位置と値を取得
            indices = tf.concat(
                [
                    tf.expand_dims(tf.range(batch_size), axis=1),  # バッチサイズ分のインデックス
                    tf.fill([batch_size, 1], i),                  # i 番目のインデックス
                    tf.fill([batch_size, 1], i)                   # i 番目のインデックス
                ],
                axis=1
            )
            updates = cost + tf.minimum(
                dtw_matrix[:batch_size, i - 1, i - 1],
                tf.minimum(dtw_matrix[:batch_size, i - 1, i], dtw_matrix[:batch_size, i, i - 1])
            )
            dtw_matrix = tf.tensor_scatter_nd_update(dtw_matrix, indices, updates)
            return dtw_matrix

        # DTW行列を計算
        for i in tf.range(1, seq_len + 1):
            dtw_matrix = dtw_step(i, dtw_matrix)

        # 実際のバッチサイズに基づき結果を抽出
        dtw_distance = dtw_matrix[:batch_size, -1, -1]
        return dtw_distance


    @property
    def metrics(self):
        return [
            self.train_loss_tracker, 
            self.val_loss_tracker, 
            self.latent_nll_tracker, 
            self.dtw_loss_tracker, 
            self.mse_loss_tracker
        ] + self.metrics_list
    
    @tf.function
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred, (latent_mean, latent_log_var, z_samples, log_qz) = self(x, return_state=True, training=True)
            
            latent_true = self.wae_encoder(y_true)
            
            latent_nll = self.compute_latent_nll_loss(latent_true, latent_mean, latent_log_var, log_qz)
            dtw_loss = self.compute_dtw_loss(y_true, y_pred)
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true[:,:3], y_pred[:,:3]))
            total_loss = (
                self.latent_nll_weight * latent_nll +
                self.dtw_weight * dtw_loss +
                self.mse_weight * mse_loss
            )
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.train_loss_tracker.update_state(total_loss)
        self.latent_nll_tracker.update_state(latent_nll)
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
        x, y_true = data
        
        # Forward pass (training=False)
        y_pred, (latent_mean, latent_log_var, z_samples, log_qz) = self(x, return_state=True)
        
        # Encode ground truth to latent space
        latent_true = self.wae_encoder(y_true)
        
        # Compute losses 
        latent_nll = self.compute_latent_nll_loss(latent_true, latent_mean, latent_log_var, log_qz)
        dtw_loss = self.compute_dtw_loss(y_true, y_pred)
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true[:,:3], y_pred[:,:3]))
        total_loss = (
            self.latent_nll_weight * latent_nll +
            self.dtw_weight * dtw_loss +
            self.mse_weight * mse_loss
        )
        
        # Update val loss trackers
        self.val_loss_tracker.update_state(total_loss)
        self.latent_nll_tracker.update_state(latent_nll)
        self.dtw_loss_tracker.update_state(dtw_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        
        # Update metrics
        for metric in self.metrics_list:
            metric.update_state(y_true, y_pred)
        
        # Return metrics
        return {
            "loss": self.val_loss_tracker.result(),
            "latent_nll": self.latent_nll_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics_list}
        }
    
    def get_config(self):
        config = super(PosePredictionModel, self).get_config()
        return config