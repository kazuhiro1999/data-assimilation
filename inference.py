import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

from dataset import FULL_BODY_KEYS, UPPER_BODY_KEYS, AssimilationDataGenerator, MotionDataLoader
from model.v2.assimilation import DeepLatentSpaceAssimilationModelV4
from utils.metrics import MPJPEError, MPJPEErrorFrame, calc_mpjpe_error


def create_inference_dataset(dataloader, filename, parameters):
    fps = parameters.get("frame_rate", 30)
    tdpt_delay_sec = parameters.get("parameters", {}).get("delay", 0.4)  # TDPT's delay
    tdpt_delay_frame = int(tdpt_delay_sec * fps)
    upper_body_indices = [FULL_BODY_KEYS.index(joint_name) for joint_name in UPPER_BODY_KEYS]        
    invalid_frames = dataloader.get_invalid_indices(filename)
                                       
    # ポーズデータの取得
    positions_true = dataloader.get_positions(filename, data_type='MultipleCameras')
    positions_iobt = dataloader.get_positions(filename, data_type='IOBT')
    positions_tdpt = dataloader.get_positions(filename, data_type='TDPT')
    
    def interpolate(array, fps, original_fps=60):
        n_frames = array.shape[0]
        end_timestamp = n_frames / original_fps

        t0 = np.linspace(0, end_timestamp, n_frames)
        g = interp1d(t0, array, axis=0, kind='linear', fill_value="extrapolate")

        t = np.arange(0, end_timestamp, (1/fps))
        new_array = g(t).astype(np.float32)

        return new_array

    positions_true = interpolate(positions_true, fps=fps, original_fps=60)
    positions_iobt = interpolate(positions_iobt, fps=fps, original_fps=60)
    positions_tdpt = interpolate(positions_tdpt, fps=fps, original_fps=60)
    
    invalid_frames = [int(f * (fps / 60)) for f in invalid_frames]

    # 位置の相対化（センタリング）
    positions_true -= positions_true[:, :1]        
    positions_iobt -= positions_iobt[:, :1]        
    positions_tdpt -= positions_tdpt[:, :1]        
        
    seq_length = min(len(positions_true), len(positions_iobt), len(positions_tdpt) - tdpt_delay_frame)      

    return positions_tdpt[tdpt_delay_frame:tdpt_delay_frame+seq_length].astype(np.float32), positions_iobt[:seq_length, upper_body_indices].astype(np.float32), positions_true[:seq_length].astype(np.float32), invalid_frames


# Inference program
def run_inference(model, dataloader, filename, parameters, expected_delay=0.1, return_inputs=False):
    context_window = parameters.get("input_seq_length", 30)
    prediction_steps = parameters.get("output_seq_length", 12)
    frame_rate = parameters.get("frame_rate", 30)

    frame_delay = min(int(expected_delay * frame_rate) - 1, 11)   # MAX_Length=12 
    
    X_tdpt, X_iobt, Y_true, invalid_frames = create_inference_dataset(dataloader, filename, parameters=parameters)
    
    # 位置の相対化（センタリング）
    root_true = Y_true[:, :1].copy()        
    root_iobt = X_iobt[:, :1].copy()        
    root_tdpt = X_tdpt[:, :1].copy() 
    
    X_iobt -= root_iobt
    X_tdpt -= root_tdpt

    # 入力データを一括取得
    indices = np.arange(context_window + frame_delay, len(Y_true) - prediction_steps + frame_delay)
    is_valid = ~np.isin(indices, invalid_frames)

    x_past_batch = np.array([X_tdpt[t - context_window - frame_delay : t - frame_delay] for t in indices])
    x_obs_batch = np.array([X_iobt[t - frame_delay : t - frame_delay + prediction_steps] for t in indices])

    # モデル予測を一括実行
    inputs = [x_past_batch, x_obs_batch]
    preds = model.predict(inputs)

    # 予測結果を取得
    y_true = Y_true[indices]  # 真値を一括取得
    y_pred = preds[:, frame_delay] + root_iobt[indices] # frame_delay の位置を取得

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if return_inputs:
        return y_true, y_pred, is_valid, (x_past_batch[:, -1] + root_tdpt[context_window:context_window+len(indices)]), (x_obs_batch[:, frame_delay] + root_iobt[indices])
        return y_true, y_pred, is_valid, x_past_batch[:, -1], x_obs_batch[:, frame_delay]
    
    return y_true, y_pred, is_valid


def main():

    # Load configuration file
    with open("experiments/config.json", "r") as f:
        config = json.load(f)

    # Paths for test data (.pkl)
    TEST_FILES = config["test_files"]

    # Dataset information
    DATASET_INFO = config["dataset_info"]

    # Setup data loader
    test_dataloader = MotionDataLoader(TEST_FILES)

    # Create dataset
    test_generator = AssimilationDataGenerator(test_dataloader)
    test_dataset = test_generator.create_dataset(DATASET_INFO)

    X_past_test, X_obs_test, Y_test = test_dataset.get('X_past'), test_dataset.get('X_obs'), test_dataset.get('Y')

    # Show dataset size
    print(f"Test Dataset X_past:{X_past_test.shape}, X_obs:{X_obs_test.shape}, Y:{Y_test.shape}")

    results = {}

    test_files = {
        "Daily": test_dataloader.get_all_filenames()[0:10],
        "Dance": test_dataloader.get_all_filenames()[10:20],
        "Sport": test_dataloader.get_all_filenames()[20:30]
    }

    # load trained model
    model_load_path = "pretrained/dlsa_light_v4_yyyymmdd.keras"
    custom_objects = {
        "DeepLatentSpaceAssimilationModelV4": DeepLatentSpaceAssimilationModelV4,
        "MPJPEError": MPJPEError,
        "MPJPEErrorFrame": MPJPEErrorFrame
    }

    model = load_model(model_load_path, custom_objects=custom_objects)
    print(print(f"model loaded from: {model_load_path}"))

    delays = [0.1, 0.2, 0.4]

    # simulate inference
    for category, files in test_files.items():    
        result = {}
        
        for filename in files:     
            result[filename] = {}
            
            for delay in delays:
                y_true, y_pred, is_valid = run_inference(model, test_dataloader, filename, parameters=DATASET_INFO, expected_delay=delay)
                mpjpe_error_infer = calc_mpjpe_error(y_true[is_valid], y_pred[is_valid]).numpy()

                print(f"{filename}: mpjpe={mpjpe_error_infer}")

                result[filename][f"mpjpe@{int(delay*1000)}ms"] = float(mpjpe_error_infer)
                
        results[category] = result


    # results
    aggregated = {}
    
    for category, actions in results.items():
        # for each delays
        delay_sums = {'mpjpe@100ms': 0.0, 'mpjpe@200ms': 0.0, 'mpjpe@400ms': 0.0}
        count = 0

        for action, delays in actions.items():
            for delay, value in delays.items():
                delay_sums[delay] += value
            count += 1

        # calc average
        aggregated[category] = {
            delay: float(delay_sums[delay]) / count for delay in delay_sums
        }

    for category, values in aggregated.items():
        print(f"{category}:")
        for delay, avg in values.items():
            print(f"  {delay}: {avg:.6f}")

    # output experimental results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_json = {
        "date": timestamp,
        "test_dataset_info": DATASET_INFO,
        "summary": aggregated,
        "results": results,
        "notes": "Observation based batch prediction model(V4). cross-attention with observational data → attention with past sequence (num_layer=1)."
    }

    os.makedirs("experiments", exist_ok=True)
    filename = f"experiments/results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=4)

    print(f"Inference results saved to: {filename}")