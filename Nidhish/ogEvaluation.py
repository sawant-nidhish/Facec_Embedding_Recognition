import sys
sys.path.insert(0,'./')
from tensorflow import keras
import evals
import tensorflow as tf
import IJB_evals
import matplotlib.pyplot as plt
import keras_cv_attention_models
import GhostFaceNets, GhostFaceNets_with_Bias

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

#Either
basic_model = tf.keras.models.load_model('checkpoints/ghost-S1-random_plus_rect_pretrained_basic_model_latest', compile=False)

ee = evals.eval_callback(basic_model, 'datasets/faces_emore/lfw.bin', batch_size=256, flip=True, PCA_acc=False)
ee.on_epoch_end(0)

ee = evals.eval_callback(basic_model, 'datasets/faces_emore/cfp_fp.bin', batch_size=256, flip=True, PCA_acc=False)
ee.on_epoch_end(0)

ee = evals.eval_callback(basic_model, 'datasets/faces_emore/agedb_30.bin', batch_size=256, flip=True, PCA_acc=False)
ee.on_epoch_end(0)
