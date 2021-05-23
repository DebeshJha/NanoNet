
import os
import json
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tensorflow.keras.optimizers import Adam
from metrics import iou, dice_coef, dice_loss, bce_dice_loss

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_model_file(path):
    with CustomObjectScope({
            'iou':iou,
            'dice_coef':dice_coef,
            'dice_loss':dice_loss,
            'bce_dice_loss': bce_dice_loss
        }):
        model = tf.keras.models.load_model(path)
        return model
