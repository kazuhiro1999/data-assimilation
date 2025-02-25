import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from utils.metrics import MPJPEError


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, l2_reg=0.01):
        super(GraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.joints_dim = input_shape[-2]
        self.A = self.add_weight(name="A",
                                 shape=(self.joints_dim, self.joints_dim),
                                 initializer=tf.keras.initializers.RandomUniform(minval=-1/math.sqrt(self.joints_dim),
                                                                                 maxval=1/math.sqrt(self.joints_dim)),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.W = self.add_weight(name="W",
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, x):
        # x shape: (batch_size, joints, channels)
        x = tf.einsum('bvc,vw->bwc', x, self.A)
        return tf.einsum('bvc,co->bvo', x, self.W)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    
    
class GCNLayer(layers.Layer):
    def __init__(self, units, activation='relu', use_batch_norm=True, use_residual=True, dropout_rate=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual        
        self.dropout_rate = dropout_rate
        
         # GraphConvolution layer
        self.graph_conv = GraphConvolution(self.units)
        
        # Initialize BatchNormalization in __init__
        self.batch_norm = keras.layers.BatchNormalization(axis=-1) if use_batch_norm else None

        # Activation function
        self.activation_fn = keras.layers.Activation(self.activation)
        
        self.dropout = keras.layers.Dropout(self.dropout_rate) if dropout_rate else None
        
        # Initialize Residual Connection
        self.residual_conv = keras.layers.Conv1D(
            filters=self.units,
            kernel_size=1,
            padding='same'
        ) if use_residual else None


    def call(self, inputs, training=False):    
        res = inputs
        
        # Apply GraphConvolution
        x = self.graph_conv(inputs)
        
        # Apply Batch Normalization (if specified)
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        
        # Apply Activation function
        x = self.activation_fn(x)
        
        if self.dropout_rate:
            x = self.dropout(x, training=training)
        
        # Apply Residual Connection (if specified)
        if self.use_residual:
            if res.shape[-1] != x.shape[-1]:
                res = self.residual_conv(res)
            # Residual connection adds the original input to the output
            x = res + x
        
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        config = super(GCNLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'use_residual': self.use_residual,
            'dropout_rate': self.dropout_rate,
        })
        return config
    

class StandardizationLayer(layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super(StandardizationLayer, self).__init__()
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        
    def build(self, input_shape):        
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)  
        flattened_inputs = tf.reshape(inputs, (shape[0], -1))  # (n_frames, n_joints*3)

        # 標準化
        normalized = (flattened_inputs - self.mean) / self.std

        # 元の形状に戻す
        return tf.reshape(normalized, shape)  # (n_frames, n_joints, 3)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    # シリアライズ用にget_configを実装
    def get_config(self):
        config = super(StandardizationLayer, self).get_config()
        config.update({
            'mean': self.mean.numpy().tolist(),
            'std': self.std.numpy().tolist()
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        mean = tf.convert_to_tensor(config.pop("mean")["config"]["value"], dtype=tf.float32)
        std = tf.convert_to_tensor(config.pop("std")["config"]["value"], dtype=tf.float32)
        return cls(mean=mean, std=std, **config)
    

class InverseStandardizationLayer(layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super(InverseStandardizationLayer, self).__init__()
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.convert_to_tensor(std, dtype=tf.float32)
        
    def build(self, input_shape):        
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        flattened_inputs = tf.reshape(inputs, (shape[0], -1))  # (n_frames, n_joints*3)

        # 逆標準化
        denormalized = (flattened_inputs * self.std) + self.mean

        # 元の形状に戻す
        return tf.reshape(denormalized, shape)  # (n_frames, n_joints, 3)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    # シリアライズ用にget_configを実装
    def get_config(self):
        config = super(InverseStandardizationLayer, self).get_config()
        config.update({
            'mean': self.mean.numpy().tolist(),
            'std': self.std.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        mean = tf.convert_to_tensor(config.pop("mean")["config"]["value"], dtype=tf.float32)
        std = tf.convert_to_tensor(config.pop("std")["config"]["value"], dtype=tf.float32)
        return cls(mean=mean, std=std, **config)
    

class WAE(keras.Model):
    def __init__(self, n_joints, n_features, latent_dim, encoder_dims, decoder_dims, dropout_rate, mean=None, std=None, 
                 reconstruction_loss_reg=1.0, mmd_loss_reg=0.1, consistency_loss_reg=0.01, **kwargs):
        super(WAE, self).__init__()
        self.n_joints = n_joints
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims  
        self.decoder_dims = decoder_dims  
        self.dropout_rate = dropout_rate
        self.mean = mean
        self.std = std
        self.reconstruction_loss_reg = reconstruction_loss_reg
        self.mmd_loss_reg = mmd_loss_reg
        self.consistency_loss_reg = consistency_loss_reg
        
        self.normalize_layer = StandardizationLayer(mean=mean, std=std)
        self.denormalize_layer = InverseStandardizationLayer(mean=mean, std=std)
        
        self.encoder = None
        self.decoder = None
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")
        self.consistency_loss_tracker = tf.keras.metrics.Mean(name="consistency_loss")
        
        self.metrics_list = [
            MPJPEError(name="mpjpe_error"),
        ]
            
    def build(self, input_shape):    
        super(WAE, self).build(input_shape)        
        
        encoder_inputs = keras.Input(shape=(self.n_joints, self.n_features))
        x = encoder_inputs
        for units in self.encoder_dims:  
            x = GCNLayer(units=units, dropout_rate=self.dropout_rate)(x)
        x = layers.GlobalAveragePooling1D()(x)
        encoder_outputs = layers.Dense(self.latent_dim, 
                               kernel_regularizer=keras.regularizers.L2(0.01), 
                               bias_regularizer=keras.regularizers.L2(0.01), 
                               activity_regularizer=keras.regularizers.L2(0.01))(x)
        self.encoder = Model(encoder_inputs, encoder_outputs, name='PoseEncoder')
        
        decoder_inputs = keras.Input(shape=(self.latent_dim,))
        x = decoder_inputs
        for units in self.decoder_dims:  
            x = layers.Dense(units, activation='relu')(x)
        x = layers.Dense(self.n_joints * self.n_features)(x)
        decoder_outputs = layers.Reshape((self.n_joints, self.n_features))(x)
        self.decoder = Model(decoder_inputs, decoder_outputs, name='PoseDecoder')
        
        self.normalize_layer.build(input_shape=(None, self.n_joints, self.n_features))
        self.denormalize_layer.build(input_shape=(None, self.latent_dim))
        
    def call(self, inputs):
        z = self.encode(inputs)
        return self.decode(z)
    
    def encode(self, x):
        x = self.normalize_layer(x)
        return self.encoder(x)
    
    def decode(self, z):
        x_recon = self.decoder(z)
        return self.denormalize_layer(x_recon)
        
       
    @tf.function
    def compute_mmd_penalty(self, sample_qz, sample_pz, kernel='RBF'):
        """Calculate MMD penalty between sample_qz and sample_pz."""
        sigma2_p = 1.0  # Assume prior p(z) ~ N(0, I), hence variance = 1
        batch_size = tf.shape(sample_qz)[0]

        # Compute pairwise distances
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if kernel == 'RBF':
            # Gaussian kernel width (sigma^2) heuristic
            sigma2_k = tf.reduce_mean(distances) / 2.0
            res1 = tf.exp(-distances_qz / (2.0 * sigma2_k))
            res2 = tf.exp(-distances_pz / (2.0 * sigma2_k))
            res3 = tf.exp(-distances / (2.0 * sigma2_k))

            stat = tf.reduce_mean(res1) + tf.reduce_mean(res2) - 2 * tf.reduce_mean(res3)
        else:
            raise ValueError("Unknown kernel")

        return stat
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.mmd_loss_tracker,
            self.consistency_loss_tracker
        ] + self.metrics_list
    
    @tf.function
    def train_step(self, data):
        x, y_true = data  # Input data      
        
        with tf.GradientTape() as tape:            
            z = self.encode(x)
            recon = self.decode(z)
            
            # Reconstruction loss (MSE)
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(x, recon))

            # Prior sample from a standard normal distribution
            pz_sample = tf.random.normal(tf.shape(z))

            # MMD loss
            mmd_loss = self.compute_mmd_penalty(z, pz_sample)
            
            # Consistency loss     
            latent_pred = self.encode(recon)
            consistency_loss = tf.reduce_mean(tf.keras.losses.mse(z, latent_pred))

            # Combined loss
            loss = self.reconstruction_loss_reg * mse_loss + self.mmd_loss_reg * mmd_loss + self.consistency_loss_reg * consistency_loss
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(mse_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        self.consistency_loss_tracker.update_state(consistency_loss)
        
        for metric in self.metrics_list:
            metric.update_state(x, recon)
        
        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
            "consistency_loss": self.consistency_loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics_list}
        }
    
    @tf.function
    def test_step(self, data):
        x, y_true = data  # Input data          
        z = self.encode(x)
        recon = self.decode(z)
        
        # Reconstruction loss (MSE)
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(x, recon))
        # Prior sample from a standard normal distribution
        pz_sample = tf.random.normal(tf.shape(z))
        # MMD loss
        mmd_loss = self.compute_mmd_penalty(z, pz_sample)
        # Consistency loss     
        latent_pred = self.encode(recon)
        consistency_loss = tf.reduce_mean(tf.keras.losses.mse(z, latent_pred))
        
        # Combined loss
        loss = self.reconstruction_loss_reg * mse_loss + self.mmd_loss_reg * mmd_loss + self.consistency_loss_reg * consistency_loss
        
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(mse_loss)
        self.mmd_loss_tracker.update_state(mmd_loss)
        self.consistency_loss_tracker.update_state(consistency_loss)
        
        for metric in self.metrics_list:
            metric.update_state(x, recon)
            
        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "mmd_loss": self.mmd_loss_tracker.result(),
            "consistency_loss": self.consistency_loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics_list}
        }
    
    def get_config(self):
        config = super(WAE, self).get_config()
        config.update({
            'n_joints': self.n_joints,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'encoder_dims': self.encoder_dims,  
            'decoder_dims': self.decoder_dims,  
            'dropout_rate': self.dropout_rate,
            'mean': self.mean,
            'std': self.std,
            'reconstruction_loss_reg': self.reconstruction_loss_reg,
            'mmd_loss_reg': self.mmd_loss_reg,
            'consistency_loss_reg': self.consistency_loss_reg
        })
        return config
