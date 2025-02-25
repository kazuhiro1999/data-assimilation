import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class PoseDatasetPreprocessor:
    def __init__(self, method='normalize', add_noise=False):
        """
        Parameters:
        - method: 'normalize' or 'standardize' to specify the preprocessing method.
        - feature_range: Tuple (min, max) for normalization. Ignored if standardization is used.
        """
        self.method = method
        self.scaler = None  # This will hold the scaler used for the dataset

    def fit(self, pose_data):
        """
        Fit the scaler to the data and transform it.
        
        Parameters:
        - pose_data: numpy array of shape (n_frames, n_joints, 3), global pose positions.
        
        Returns:
        - Transformed pose_data.
        """
        input_shape = pose_data.shape
        
        if pose_data.ndim > 3:
            pose_data = pose_data.reshape([-1, input_shape[-2], input_shape[-1]])
            
        n_frames, n_joints, n_features = pose_data.shape
        
        # Reshape data for scaler to work on all features
        pose_data_reshaped = pose_data.reshape(n_frames, -1)  # shape (n_frames, n_joints * n_features)
        
        if self.method == 'normalize':
            # Use MinMaxScaler for normalization
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.method == 'standardize':
            # Use StandardScaler for standardization
            self.scaler = StandardScaler()
        else:
            raise ValueError("Invalid method. Choose 'normalize' or 'standardize'.")
        
        # Fit the data
        self.scaler.fit(pose_data_reshaped)
        
        return self.scaler
    
    def transform(self, pose_data):
        """
        Fit the scaler to the data and transform it.
        
        Parameters:
        - pose_data: numpy array of shape (n_frames, n_joints, 3), global pose positions.
        
        Returns:
        - Transformed pose_data.
        """
        input_shape = pose_data.shape
        
        if pose_data.ndim > 3:
            pose_data = pose_data.reshape([-1, input_shape[-2], input_shape[-1]])
            
        n_frames, n_joints, n_features = pose_data.shape
        
        # Reshape data for scaler to work on all features
        pose_data_reshaped = pose_data.reshape(n_frames, -1)  # shape (n_frames, n_joints * n_features)
        
        # Fit and transform the data
        pose_data_scaled = self.scaler.transform(pose_data_reshaped)
        
        # Reshape back to original shape (n_frames, n_joints, 3)
        pose_data_scaled = pose_data_scaled.reshape(input_shape)
        
        return pose_data_scaled.astype(np.float32)

    def inverse_transform(self, scaled_pose_data):
        """
        Inverse the transformation to get back the original data.
        
        Parameters:
        - scaled_pose_data: numpy array of shape (n_frames, n_joints, 3), transformed pose data.
        
        Returns:
        - Original pose data after inverse transformation.
        """
        input_shape = scaled_pose_data.shape
        
        if scaled_pose_data.ndim > 3:
            scaled_pose_data = scaled_pose_data.reshape([-1, input_shape[-2], input_shape[-1]])
            
        n_frames, n_joints, n_features = scaled_pose_data.shape
        
        # Reshape to match the original scaler input format
        scaled_pose_data_reshaped = scaled_pose_data.reshape(n_frames, -1)  # shape (n_frames, n_joints * 3)
        
        # Inverse transform
        pose_data_original = self.scaler.inverse_transform(scaled_pose_data_reshaped)
        
        # Reshape back to (n_frames, n_joints, 3)
        pose_data_original = pose_data_original.reshape(input_shape)
        
        return pose_data_original.astype(np.float32)
