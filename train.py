
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from glob import glob
from sklearn.model_selection import train_test_split
from model import NanoNet_A, NanoNet_B, NanoNet_C
from utils import shuffling, create_dir
from metrics import dice_loss, dice_coef, iou, bce_dice_loss
from data import tf_dataset, load_data

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Remove folders and files """
    # os.system("rm files/files.csv")
    # os.system("rm -r logs")

    """ Hyperparameters """
    input_shape = (256, 256, 3)
    batch_size = 8
    lr = 1e-4
    epochs = 200
    model_name = "A"
    model_path = f"files/{model_name}/model.h5"
    csv_path = f"files/{model_name}/model.csv"
    log_path = f"logs/{model_name}/"

    """ Creating folders """
    create_dir(f"files/{model_name}")

    """ Dataset """
    path = "/../../Dataset/Kvasir-SEG/"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    if model_name == "A":
        model = NanoNet_A(input_shape)
    elif model_name == "B":
        model = NanoNet_B(input_shape)
    elif model_name == "C":
        model = NanoNet_C(input_shape)

    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(log_dir=log_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)
