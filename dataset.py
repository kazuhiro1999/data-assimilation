import pickle
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


DATA_TYPES = ['MultipleCameras', 'IOBT', 'TDPT']

FULL_BODY_KEYS = ['Hips','Chest','Neck','Head','LeftUpperArm','LeftLowerArm','LeftHand','RightUpperArm','RightLowerArm','RightHand','LeftUpperLeg','LeftLowerLeg','LeftFoot','LeftToes','RightUpperLeg','RightLowerLeg','RightFoot','RightToes']
UPPER_BODY_KEYS = ['Hips','Chest','Neck','Head','LeftUpperArm','LeftLowerArm','LeftHand','RightUpperArm','RightLowerArm','RightHand']
LOWER_BODY_KEYS = ['Hips','LeftUpperLeg','LeftLowerLeg','LeftFoot','LeftToes','RightUpperLeg','RightLowerLeg','RightFoot','RightToes']

KINEMATIC_TREE = [
    ["Hips", "Chest", "Neck", "Head"],
    ["Hips", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"],
    ["Hips", "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"],
    ["Neck", "LeftUpperArm", "LeftLowerArm", "LeftHand"],
    ["Neck", "RightUpperArm", "RightLowerArm", "RightHand"],
]


class MotionDataLoader:
    def __init__(self, file_paths):
        """
        :param file_paths: list of dataset file path
        """
        self.data = {}
        for path in file_paths:
            with open(path, 'rb') as p:
                self.data.update(pickle.load(p))
                
        print(f"{len(self.get_all_filenames())} files found.")
                
    def get_all_filenames(self):
        return list(self.data.keys())
    
    def get_positions(self, filename, data_type="MultipleCameras"):
        """
        :param filename: key
        :param data_type: "MultipleCameras", "IOBT", "TDPT" 
        """
        return self.data[filename][data_type]['positions']
    
    
    def get_invalid_indices(self, filename, data_type="MultipleCameras"):
        """
        :param filename: key
        :param data_type: "MultipleCameras", "IOBT", "TDPT" 
        """
        return self.data[filename][data_type].get('invalid_indices', [])    
    
    
class BaseDataGenerator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def create_dataset(self, dataset_info):
        raise NotImplementedError("Subclasses must implement create_dataset method")
        
    def interpolate(self, array, fps, original_fps=60):
        n_frames = array.shape[0]
        end_timestamp = n_frames / original_fps

        t0 = np.linspace(0, end_timestamp, n_frames)
        g = interp1d(t0, array, axis=0, kind='linear', fill_value="extrapolate")

        t = np.arange(0, end_timestamp, (1/fps))
        new_array = g(t).astype(np.float32)

        return new_array
    
    def align_center(self, positions):
        root_positions = positions[:, :1].copy()  # Index 0: Hips
        return positions - root_positions
    
    def rotate_y_axis(self, positions, angle_deg):
        angle_rad = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
        rotated_positions = np.dot(positions, rotation_matrix)
        return rotated_positions

    def random_scale_xyz(self, array, s_min=0.75, s_max=1.2):
        s = np.random.uniform(s_min, s_max)
        return s * array

    def add_random_noise(self, array, std=0.01):
        noise = np.random.normal(loc=0, scale=std, size=array.shape)
        return array + noise
        

class PoseDataGenerator(BaseDataGenerator):
    
    def create_dataset(self, dataset_info):
        files = self.dataloader.get_all_filenames()
        validation_split = dataset_info.get('validation_split', 0.0)
        fps = dataset_info.get('frame_rate', 30)
        sampling_rate = dataset_info.get('sampling_rate', 1)
        
        enable_centering = dataset_info['preprocessing'].get('centering', False)
        augmentation = dataset_info["preprocessing"].get("augmentation", {})
        rotation_range = augmentation.get('rotation_range', [0])
        scaling_range = augmentation.get('scaling_range', [1.0, 1.0])
        noise_std_dev = augmentation.get('noise_std_dev', 0.0)

        X, Y = [], []
        X_sub, Y_sub = [], []

        for filename in tqdm(files):            
            positions = self.dataloader.get_positions(filename)
            positions = self.interpolate(positions, fps=fps, original_fps=60)
            invalid_frames = self.dataloader.get_invalid_indices(filename)
            invalid_frames = [int(f * (fps / 60)) for f in invalid_frames]
            
            valid_indices = list(set(range(len(positions))) - set(invalid_frames))
            positions = positions[valid_indices]  # 有効フレームのみ取得
            
            if enable_centering:
                positions = self.align_center(positions)            
              
            for deg in rotation_range:              
                input_seq = self.rotate_y_axis(positions.copy(), deg)
                
                for t in range(0, len(input_seq), sampling_rate):                
                    pose = input_seq[t].copy()               
                    pose = self.random_scale_xyz(pose, *scaling_range)
                    pose = self.add_random_noise(pose, noise_std_dev)

                    if t <= len(input_seq) * (1 - validation_split):
                        X.append(pose)
                        Y.append(pose)                        
                    else:
                        X_sub.append(pose)
                        Y_sub.append(pose)  

        return {
            'X': np.array(X, dtype=np.float32), 
            'Y': np.array(Y, dtype=np.float32),
            'X_sub': np.array(X_sub, dtype=np.float32), 
            'Y_sub': np.array(Y_sub, dtype=np.float32)
        }


class PredictionDataGenerator(BaseDataGenerator):
    
    def create_dataset(self, dataset_info):
        files = self.dataloader.get_all_filenames()
        validation_split = dataset_info.get('validation_split', 0.0)
        
        input_seq_length = dataset_info.get('input_seq_length', 10)
        output_seq_length = dataset_info.get('output_seq_length', 10)
        fps = dataset_info.get('frame_rate', 30)
        
        enable_centering = dataset_info['preprocessing'].get('centering', False)
        augmentation = dataset_info["preprocessing"].get("augmentation", {})
        rotation_range = augmentation.get('rotation_range', [0])
        scaling_range = augmentation.get('scaling_range', [1.0, 1.0])
        noise_std_dev = augmentation.get('noise_std_dev', 0.0)
        stride = dataset_info.get('stride', 3)  # テスト用は1、訓練用は3など
        
        X, Y = [], []
        X_sub, Y_sub = [], []
        
        # 訓練データの処理
        for filename in tqdm(files):
            # ポーズデータの取得
            positions = self.dataloader.get_positions(filename)
            
            # フレーム補間
            positions = self.interpolate(positions, fps=fps, original_fps=60)
            
            # センタリング処理
            if enable_centering:
                positions = self.align_center(positions)
            
            seq_length = len(positions)
            max_size = seq_length - (input_seq_length + output_seq_length) + 1
            
            if max_size <= 0:
                continue  # シーケンスが短すぎる場合はスキップ
                
            for t in range(input_seq_length + output_seq_length, seq_length, stride):
                for deg in rotation_range:
                    full_seq = positions[t - input_seq_length - output_seq_length : t].copy()
                    
                    # data augumentation
                    full_seq = self.rotate_y_axis(full_seq, deg)
                    full_seq = self.random_scale_xyz(full_seq, *scaling_range)
                    full_seq = self.add_random_noise(full_seq, noise_std_dev)
                    
                    # split sequence
                    input_seq = full_seq[:input_seq_length]
                    output_seq = full_seq[input_seq_length:]
                                        
                    # 訓練データとバリデーションデータの分割
                    current_position = t - (input_seq_length + output_seq_length)
                    if current_position <= max_size * (1 - validation_split):
                        X.append(input_seq)
                        Y.append(output_seq)
                    else:
                        X_sub.append(input_seq)
                        Y_sub.append(output_seq)
            
        return {
            'X': np.array(X, dtype=np.float32), 
            'Y': np.array(Y, dtype=np.float32),
            'X_sub': np.array(X_sub, dtype=np.float32), 
            'Y_sub': np.array(Y_sub, dtype=np.float32),
        }
    
    
class AssimilationDataGenerator(BaseDataGenerator):
    
    def create_dataset(self, dataset_info):
        files = self.dataloader.get_all_filenames()
        validation_split = dataset_info.get('validation_split', 0.0)
        
        input_seq_length = dataset_info.get('input_seq_length', 10)
        output_seq_length = dataset_info.get('output_seq_length', 10)
        fps = dataset_info.get('frame_rate', 30)
        
        enable_centering = dataset_info['preprocessing'].get('centering', False)
        augmentation = dataset_info["preprocessing"].get("augmentation", {})
        rotation_range = augmentation.get('rotation_range', [0])
        scaling_range = augmentation.get('scaling_range', [1.0, 1.0])
        noise_std_dev = augmentation.get('noise_std_dev', 0.0)
        stride = dataset_info.get('stride', 3)  
        
        tdpt_delay = int(dataset_info["parameters"].get("delay", 0.0) * fps)
        upper_body_indices = [FULL_BODY_KEYS.index(joint) for joint in UPPER_BODY_KEYS]

        X_past, X_obs, Y_true = [], [], []
        X_past_sub, X_obs_sub, Y_true_sub = [], [], []

        for filename in tqdm(files):
            positions_tdpt = self.dataloader.get_positions(filename, data_type='TDPT')
            positions_iobt = self.dataloader.get_positions(filename, data_type='IOBT')
            positions_true = self.dataloader.get_positions(filename, data_type='MultipleCameras')

            positions_tdpt = self.interpolate(positions_tdpt, fps=fps, original_fps=60)
            positions_iobt = self.interpolate(positions_iobt, fps=fps, original_fps=60)
            positions_true = self.interpolate(positions_true, fps=fps, original_fps=60)

            if enable_centering:
                positions_tdpt = self.align_center(positions_tdpt)
                positions_iobt = self.align_center(positions_iobt)
                positions_true = self.align_center(positions_true)
                
            seq_length = min(len(positions_true), len(positions_iobt), len(positions_tdpt) - tdpt_delay)
            max_size = 1 + int((seq_length - (input_seq_length + output_seq_length)))
            
            if max_size <= 0:
                continue  # シーケンスが短すぎる場合はスキップ

            for t in range(input_seq_length + output_seq_length, seq_length, stride):
                for deg in rotation_range:
                    
                    input_past_seq = positions_tdpt[t + tdpt_delay - input_seq_length - output_seq_length : t + tdpt_delay - output_seq_length].copy()
                    input_obs_seq = positions_iobt[t - output_seq_length : t, upper_body_indices].copy()
                    output_seq = positions_true[t - output_seq_length : t].copy()

                    input_past_seq = self.rotate_y_axis(input_past_seq, deg)
                    input_obs_seq = self.rotate_y_axis(input_obs_seq, deg)
                    output_seq = self.rotate_y_axis(output_seq, deg)

                    scale_factor = np.random.uniform(*scaling_range)
                    input_past_seq *= scale_factor
                    input_obs_seq *= scale_factor
                    output_seq *= scale_factor

                    input_past_seq = self.add_random_noise(input_past_seq, noise_std_dev)
                    input_obs_seq = self.add_random_noise(input_obs_seq, noise_std_dev)
                    output_seq = self.add_random_noise(output_seq, noise_std_dev) 

                    current_position = t - (input_seq_length + output_seq_length)
                    if current_position <= max_size * (1 - validation_split):
                        X_past.append(input_past_seq)
                        X_obs.append(input_obs_seq)
                        Y_true.append(output_seq)                        
                    else:
                        X_past_sub.append(input_past_seq)
                        X_obs_sub.append(input_obs_seq)
                        Y_true_sub.append(output_seq)  

        return {
            'X_past': np.array(X_past, dtype=np.float32), 
            'X_obs': np.array(X_obs, dtype=np.float32), 
            'Y': np.array(Y_true, dtype=np.float32),
            'X_past_sub': np.array(X_past_sub, dtype=np.float32), 
            'X_obs_sub': np.array(X_obs_sub, dtype=np.float32), 
            'Y_sub': np.array(Y_true_sub, dtype=np.float32)
        }
