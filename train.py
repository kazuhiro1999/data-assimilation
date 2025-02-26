import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from dataset import MotionDataLoader, AssimilationDataGenerator, FULL_BODY_KEYS
from model.wae import WAE
from model.time_stepping import VariationalTimeSteppingTransformerNF
from model.assimilation import DeepLatentSpaceAssimilationModel
from utils.metrics import calc_mpjpe_error
from utils.visualization import draw_3d_pose


# シード固定
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


if __name__ == '__main__':

    # 設定ファイルの読み込み
    with open("experiments/config.json", "r") as f:
        config = json.load(f)

    # 訓練・テストデータのパス
    TRAIN_FILES = config["train_files"]
    TEST_FILES = config["test_files"]

    # データセット情報
    DATASET_INFO = config["dataset_info"]

    # ハイパーパラメータ
    batch_size = config["batch_size"]
    latent_dim = config["latent_dim"]
    num_joints = config["num_joints"]
    num_upperbody_joints = config["num_upperbody_joints"]
    epochs = config["epochs"]

    # コンテキストウィンドウと予測ステップ
    context_window = DATASET_INFO["input_seq_length"]
    prediction_steps = DATASET_INFO["output_seq_length"]

    # データローダー作成
    train_dataloader = MotionDataLoader(TRAIN_FILES)
    test_dataloader = MotionDataLoader(TEST_FILES)

    # データセット作成
    train_generator = AssimilationDataGenerator(train_dataloader)
    train_dataset = train_generator.create_dataset(DATASET_INFO)

    X_past_train, X_obs_train, Y_train = train_dataset.get('X_past'), train_dataset.get('X_obs'), train_dataset.get('Y')
    X_past_valid, X_obs_valid, Y_valid = train_dataset.get('X_past_sub'), train_dataset.get('X_obs_sub'), train_dataset.get('Y_sub')

    # データセットのサイズ表示
    print(f"Train Dataset X_past:{X_past_train.shape}, X_obs:{X_obs_train.shape}, Y:{Y_train.shape}")
    print(f"Validation Dataset X_past:{X_past_valid.shape}, X_obs:{X_obs_valid.shape}, Y:{Y_valid.shape}")

    # 可視化 (プロットを画像として保存)
    os.makedirs("output", exist_ok=True)
    n = 6000
    j = FULL_BODY_KEYS.index('RightHand')

    plt.figure()
    plt.plot(np.arange(context_window), X_past_train[n,:,j], color='green')
    plt.plot(np.arange(context_window, context_window + prediction_steps), X_obs_train[n,:,j], ":", color='blue')
    plt.plot(np.arange(context_window, context_window + prediction_steps), Y_train[n,:,j], color='red')
    plt.savefig("output/training_data_plot.png")
    plt.close()

    # 事前学習済みモデルの読み込み
    wae_model = keras.models.load_model("pretrained/pose_autoencoder.keras", custom_objects={"WAE": WAE})  
    time_stepping_model = keras.models.load_model("pretrained/time_stepping_transformer.keras", custom_objects={"VariationalTimeSteppingTransformerNF": VariationalTimeSteppingTransformerNF})

    # モデル構築
    model = DeepLatentSpaceAssimilationModel(
        wae=wae_model, 
        time_stepping_model=time_stepping_model, 
        latent_dim=latent_dim, 
        context_window=context_window, 
        prediction_steps=prediction_steps,
        num_joints=num_joints
    )

    model.build(input_shape=(None, context_window, num_joints, 3))

    # ダミーデータでモデルビルド確認
    dummy_input = [tf.random.uniform((1, context_window, num_joints, 3)), 
                   tf.random.uniform((1, prediction_steps, num_upperbody_joints, 3))]
    dummy_output = model(dummy_input)

    #print(dummy_output.shape)
    #model.summary()

    # モデルコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True),
    )

    # コールバック
    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/dlsa_model.keras", 
                                           monitor="val_mpjpe_error", save_best_only=True, save_weights_only=False, mode="min")
    ]

    # 学習実行
    print("Training started...")
    history = model.fit(
        [X_past_train, X_obs_train], Y_train, 
        validation_data=([X_past_valid, X_obs_valid], Y_valid), 
        batch_size=batch_size, 
        epochs=epochs, 
        shuffle=True, 
        verbose=1,  # tqdmは不要
        callbacks=callbacks
    )

    # 学習曲線のプロットと保存
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

    # モデルの保存
    model_save_path = f"pretrained/dlsa_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
