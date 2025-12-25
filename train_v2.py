import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from dataset import MotionDataLoader, AssimilationDataGenerator, FULL_BODY_KEYS
from model.wae import WAE
from model.v2.assimilation import DeepLatentSpaceAssimilationModelV4
from utils.metrics import calc_mpjpe_error
from utils.visualization import draw_3d_pose


# Fix seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


if __name__ == '__main__':

    # Load configuration file
    with open("experiments/config.json", "r") as f:
        config = json.load(f)

    # Paths for training and test data (.pkl)
    TRAIN_FILES = config["train_files"]
    TEST_FILES = config["test_files"]

    # Dataset information
    DATASET_INFO = config["dataset_info"]

    # Hyperparameters
    context_window = DATASET_INFO.get("input_seq_length")
    prediction_steps = DATASET_INFO.get("output_seq_length")
    num_joints = config.get("num_joints")
    num_upperbody_joints = config.get("num_upperbody_joints")
    latent_dim = config.get("latent_dim")
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 1e-4)
    epochs = config.get("epochs", 10)

    wae_model_path = config.get("wae_model_path")
    time_stepping_model_path = config.get("time_stepping_model_path")

    # Setup data loader
    train_dataloader = MotionDataLoader(TRAIN_FILES)
    test_dataloader = MotionDataLoader(TEST_FILES)

    # Create dataset
    train_generator = AssimilationDataGenerator(train_dataloader)
    train_dataset = train_generator.create_dataset(DATASET_INFO)

    X_past_train, X_obs_train, Y_train = train_dataset.get('X_past'), train_dataset.get('X_obs'), train_dataset.get('Y')
    X_past_valid, X_obs_valid, Y_valid = train_dataset.get('X_past_sub'), train_dataset.get('X_obs_sub'), train_dataset.get('Y_sub')

    print(f"Train Dataset X_past:{X_past_train.shape}, X_obs:{X_obs_train.shape}, Y:{Y_train.shape}")
    print(f"Validation Dataset X_past:{X_past_valid.shape}, X_obs:{X_obs_valid.shape}, Y:{Y_valid.shape}")

    # Visualization (save example data as image)
    os.makedirs("output", exist_ok=True)
    n = 6000
    j = FULL_BODY_KEYS.index('RightHand')

    plt.figure()
    plt.plot(np.arange(context_window), X_past_train[n,:,j], color='green')
    plt.plot(np.arange(context_window, context_window + prediction_steps), X_obs_train[n,:,j], ":", color='blue')
    plt.plot(np.arange(context_window, context_window + prediction_steps), Y_train[n,:,j], color='red')
    plt.savefig("output/training_data_plot.png")
    plt.close()

    # Load pre-trained models
    wae_model = keras.models.load_model(wae_model_path, custom_objects={"WAE": WAE})  

    # Build model
    model = DeepLatentSpaceAssimilationModelV4(
        wae=wae_model, 
        latent_dim=latent_dim, 
        context_window=context_window, 
        prediction_steps=prediction_steps,
        num_joints=num_joints,    
        num_flows=3,
        obs_encoder_dims=[512, 256],
        obs_dropout_rate=0.2,
        pred_dim=256,
        pred_num_heads=4,
        pred_num_blocks=1,
        pred_dropout_rate=0.2,
        nll_weight=0.01,
        dtw_weight=1.0,
        mse_weight=100.0,
    )

    # Check model with dummy data
    dummy_input = [tf.random.uniform((1, context_window, num_joints, 3)), 
                   tf.random.uniform((1, prediction_steps, num_upperbody_joints, 3))]
    dummy_output = model(dummy_input)

    #print(dummy_output.shape)
    #model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True),
    )

    # Callbacks
    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/dlsa_model.keras", 
                                           monitor="val_mpjpe_error", save_best_only=True, save_weights_only=False, mode="min")
    ]

    # Start training
    print("Training started...")
    history = model.fit(
        [X_past_train, X_obs_train], Y_train, 
        validation_data=([X_past_valid, X_obs_valid], Y_valid), 
        batch_size=batch_size, 
        epochs=epochs, 
        shuffle=True, 
        verbose=1,  
        callbacks=callbacks
    )

    # Plot and save learning curves
    metrics = ["dtw_loss", "latent_nll", "loss", "mpjpe_error"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    for i in range(2):
        for j in range(2):
            n = 2 * i + j
            ax[i][j].plot(history.history[metrics[n]], label=metrics[n])
            ax[i][j].plot(history.history[f'val_{metrics[n]}'], label=f'val_{metrics[n]}', color='red')
            ax[i][j].legend()
    plt.savefig("output/training_history.png")
    plt.close()

    # Save model
    model_save_path = f"pretrained/dlsa_light_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
