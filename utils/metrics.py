import tensorflow as tf


def calc_mpjpe_error(batch_pred, batch_gt):
    batch_pred = tf.reshape(batch_pred, [-1, 3])
    batch_gt = tf.reshape(batch_gt, [-1, 3])

    return tf.reduce_mean(tf.norm(batch_gt - batch_pred, ord=2, axis=1))


def calc_mpjpe_frame(batch_pred, batch_gt, frame_idx):
    """
    Calculate cumulative MPJPE up to a specific frame index.
    :param batch_pred: Predicted poses, shape (batch_size, seq_len, num_joints, 3)
    :param batch_gt: Ground truth poses, shape (batch_size, seq_len, num_joints, 3)
    :param frame_idx: Index of the last frame to include in the calculation (inclusive)
    :return: Cumulative MPJPE up to the specified frame
    """
    seq_len = batch_pred.shape[1]
    if frame_idx >= seq_len:
        frame_idx = seq_len

    batch_pred = tf.reshape(batch_pred[:, :frame_idx + 1], [-1, 3])
    batch_gt = tf.reshape(batch_gt[:, :frame_idx + 1], [-1, 3])

    return tf.reduce_mean(tf.norm(batch_gt - batch_pred, ord=2, axis=1))


class MPJPEError(tf.keras.metrics.Metric):
    def __init__(self, name="mpjpe_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = calc_mpjpe_error(y_pred, y_true)  # 定義済みの関数を使用
        batch_size = tf.shape(y_true)[0]
        self.total_error.assign_add(error * tf.cast(batch_size, tf.float32))
        self.count.assign_add(tf.cast(batch_size, tf.float32))  # バッチサイズを加算

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)
        

class MPJPEErrorFrame(tf.keras.metrics.Metric):
    def __init__(self, frame_idx, name="mpjpe_frame", **kwargs):
        super().__init__(name=f"mpjpe@{frame_idx}frames", **kwargs)
        self.frame_idx = frame_idx
        self.total_error = self.add_weight(name="total_error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        error = calc_mpjpe_frame(y_pred, y_true, frame_idx=self.frame_idx)  # Defined function
        self.total_error.assign_add(tf.reduce_sum(error) * tf.cast(batch_size, tf.float32))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self):
        return self.total_error / self.count

    def reset_state(self):
        self.total_error.assign(0)
        self.count.assign(0)
