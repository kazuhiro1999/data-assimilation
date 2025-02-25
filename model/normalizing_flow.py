import tensorflow as tf
from tensorflow.keras import layers

# ================================
# 1. Normalizing Flow の基本クラス
# ================================
class BaseFlow(tf.keras.layers.Layer):
    """ Normalizing Flow の基本クラス """
    def __init__(self, latent_dim, **kwargs):
        super(BaseFlow, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, z):
        """ 入力 z に対して flow 変換を適用 """
        raise NotImplementedError("Subclasses must implement `call` method.")
        
    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0],)]
    
    def get_config(self):
        config = super(PlanarFlow, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config
        
# ================================
# 2. Planar Flow の実装
# ================================
class PlanarFlow(BaseFlow):
    """ Planar Flow: シンプルな潜在変換 """
    def __init__(self, latent_dim, **kwargs):
        super(PlanarFlow, self).__init__(latent_dim, **kwargs)
        self.w = self.add_weight(shape=(latent_dim, 1), initializer="random_normal", trainable=True)
        self.u = self.add_weight(shape=(latent_dim, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
        
    def build(self, input_shape):
        """ PlanarFlow の重みの初期化 """
        super(PlanarFlow, self).build(input_shape)

    def call(self, z):
        """ z に Planar Flow 変換を適用 """
        linear = tf.matmul(z, self.w) + self.b
        activation = tf.math.tanh(linear)
        flow_z = z + tf.matmul(activation, tf.transpose(self.u))
        log_det_jacobian = tf.reduce_sum(1 - tf.square(tf.math.tanh(linear)), axis=-1)
        return flow_z, log_det_jacobian
    
    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0],)]
    
    def get_config(self):
        config = super(PlanarFlow, self).get_config()
        return config

# ================================
# 3. Invertible Flow (Affine Coupling Layer)
# ================================
class InvertibleFlow(BaseFlow):
    """ シンプルな Affine Coupling Layer を使った Invertible Flow """
    def __init__(self, latent_dim, **kwargs):
        super(InvertibleFlow, self).__init__(latent_dim, **kwargs)
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.scale_nn = tf.keras.Sequential([
            layers.Dense(self.half_dim * 4, activation='relu'),
            layers.Dense(self.half_dim, activation='tanh')  # スケールパラメータ
        ])
        self.shift_nn = tf.keras.Sequential([
            layers.Dense(self.half_dim * 4, activation='relu'),
            layers.Dense(self.half_dim)  # シフトパラメータ
        ])
    
    def build(self, input_shape):
        """ InvertibleFlow の重みの初期化 """
        super(InvertibleFlow, self).build(input_shape)
        self.scale_nn.build(input_shape=input_shape[:-1] + (self.half_dim,))
        self.shift_nn.build(input_shape=input_shape[:-1] + (self.half_dim,))

    def call(self, z):
        """ z に Invertible Flow 変換を適用 """
        # 入力を2つに分割
        z0, z1 = tf.split(z, [self.half_dim, self.half_dim], axis=-1)
        
        scale = self.scale_nn(z1)
        shift = self.shift_nn(z1)
        z0 = z0 * tf.exp(scale) + shift  # Affine Coupling
        
        flow_z = tf.concat([z0, z1], axis=-1)
        log_det_jacobian = tf.reduce_sum(scale, axis=-1)        
        return flow_z, log_det_jacobian
    
    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0],)]
    
    def get_config(self):
        config = super(InvertibleFlow, self).get_config()
        return config

# ================================
# 4. Normalizing Flow (複数の Flow を適用)
# ================================
class NormalizingFlow(tf.keras.layers.Layer):
    """ 複数の Flow を適用するクラス """
    def __init__(self, flows, **kwargs):
        super(NormalizingFlow, self).__init__(**kwargs)
        self.flows = flows  # Flow のリスト
        
    def build(self, input_shape):
        super(NormalizingFlow, self).build(input_shape)
        for flow in self.flows:
            flow.build(input_shape)

    def call(self, z):
        """ Flow を順番に適用 """
        log_qz = 0.0
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_qz += log_det_jacobian
        return z, log_qz
    
    def compute_output_shape(self, input_shape):
        return [input_shape, (input_shape[0],)]
    
    def get_config(self):
        config = super(NormalizingFlow, self).get_config()
        config.update({"flows": self.flows})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)