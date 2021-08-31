#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

# tf.logging.set_verbosity(tf.logging.DEBUG)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys

module_utils = os.path.join(os.getcwd(), 'utils')
sys.path.append(module_utils)
from utils.dataset_baseline import DanceSeqHDF5
from utils.plot_result import test_draw, draw_image, draw
from motion_transform import reverse_motion_transform

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
from networks.baseline import Dancer

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#############################
# Settings
#############################

BATCH_SIZE = 128
time_steps = 50

exp = 'exp_baseline'
model_name = os.path.join(exp, 'baseline_melspectro')

data_path = 'data'
configuration = {'file_pos_minmax': 'data/pos_minmax.h5',
                 'normalization': 'interval',
                 'rng_pos': [-0.9, 0.9]}

if not os.path.isdir(exp):
    os.makedirs(exp)
if not os.path.isdir(model_name):
    os.makedirs(model_name)

################################
# Loading datasets (Train/Test)
################################

train_path = os.path.join(data_path, 'train')
train_generator = DanceSeqHDF5(folder=data_path, sequence=train_path, stage='train', init_step=1)

test_path = os.path.join(data_path, 'test')
test_generator = DanceSeqHDF5()

#############################
# Helper functions
#############################

"""
def reconstruct_sequence(model, test_sequence, exp, export_to_file=True, name='test_sequence'):
    predictions = model.predict_on_batch(test_sequence).numpy()
    predictions = np.reshape(predictions, (time_steps, 69))
    predictions = reverse_motion_transform(predictions, configuration)
    predictions = np.reshape(predictions, (time_steps, 23, 3))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    name = os.path.join(exp, name)
    videoWriter = cv2.VideoWriter(name + '.avi', fourcc, 25, (600, 400))
    draw(predictions, export_to_file=export_to_file, videoWriter_enable=videoWriter)
    videoWriter.release()
    cv2.destroyAllWindows()
"""


def draw_sequence(test_sequence, exp_folder, export_to_file=True, name='sequence'):
    size = test_sequence.shape[0]
    test_sequence = np.reshape(test_sequence, (size, 23, 3))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    name = os.path.join(exp_folder, name)
    videoWriter = cv2.VideoWriter(name + '.avi', fourcc, 25, (600, 400))
    draw(test_sequence, export_to_file=export_to_file, videoWriter_enable=videoWriter)
    videoWriter.release()
    cv2.destroyAllWindows()


def draw_sequece_from_distributions(initial_frame, distributions, nom):
    sample_sequence = model.sample(initial_frame, distributions)
    sample_sequence = np.squeeze(sample_sequence)
    sample_sequence = reverse_motion_transform(sample_sequence, configuration)
    draw_sequence(sample_sequence, exp_folder=model_name, name=nom)


def plot_and_save_loss(train_loss, name, val_loss=None):
    plt.plot(train_loss)
    if val_loss is not None:
        plt.plot(val_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
    else:
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(name, 'loss_history.png'))


#############################
# Models/training
#############################

epochs = 50

batch_size = BATCH_SIZE

initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50], [1e-4, 1e-5])
optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_metric = tf.keras.metrics.Mean()
folder_images = os.path.join(exp, 'folder_image')
if not os.path.isdir(folder_images):
    os.makedirs(folder_images)

groundtruth_0 = train_generator.__getitem__(0)[1][0]
groundtruth_0 = np.reshape(groundtruth_0, (1, time_steps, y_dim))
groundtruth_0 = np.squeeze(groundtruth_0)
groundtruth_0 = reverse_motion_transform(groundtruth_0, configuration)
nom = 'groundtruth_sequence'
draw_sequence(groundtruth_0, exp_folder=model_name, name=nom)

train_batch = train_generator.__getitem__(0)[0][0]
first_frame = np.reshape(train_batch, (1, time_steps, x_dim))

nom = 'sequence_at_epoch_{:04d}.png'.format(0)
test_mu, test_sig = model.call(first_frame, train=False)
sequence_to_draw = sampler.sample_sequence([test_mu, test_sig])
sequence_to_draw = np.squeeze(sequence_to_draw)
sequence_to_draw = reverse_motion_transform(sequence_to_draw, configuration)
draw_sequence(sequence_to_draw, exp_folder=model_name, name=nom)

epochs_list = list()
loss_list = list()
# val_loss_list = list()
for epoch in range(1, epochs + 1):
    print("Start of epoch %d" % (epoch,))
    start_time = time.time()
    for step in range(train_generator.__len__()):
        example_index = step * batch_size
        train_batch = train_generator.__getitem__(example_index)
        train_x = np.reshape(train_batch[0], (BATCH_SIZE, time_steps, x_dim))
        train_y = train_batch[1]
        with tf.GradientTape() as tape:
            theta_mu, theta_sig = model(train_x)
            reconstruction = sampler.sample_sequence([theta_mu, theta_sig])
            loss = Gaussian(train_y, theta_mu, theta_sig) + tf.losses.mae(train_y, reconstruction)
            loss += model.losses

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer1.apply_gradients(zip(grads, model.trainable_weights))

        loss_metric(loss)
        if step % 100 == 0:
            output = sampler.sample_sequence([theta_mu, theta_sig])
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
            print("train input batch of shape {} and mean {}".format(train_x.shape, np.mean(train_x)))
            print("train groundtruth batch of shape {} and mean {}".format(train_y.shape, np.mean(train_y)))
            print("model output  batch of shape {} and mean {}".format(output.shape, np.mean(output)))
            print("\n")

    end_time = time.time()

    if epoch % 1 == 0:
        print('Epoch: {}, train: {}, ''time elapse for current epoch {}'.format(epoch, loss_metric.result(),
                                                                                end_time - start_time))
        print("\n")
        nom = 'sequence_at_epoch_{:04d}.png'.format(epoch)
        test_mu, test_sig = model.call(first_frame, train=False)
        sequence_to_draw = Sampling().sample_sequence([test_mu, test_sig])
        sequence_to_draw = np.squeeze(sequence_to_draw)
        sequence_to_draw = reverse_motion_transform(sequence_to_draw, configuration)
        draw_sequence(sequence_to_draw, exp_folder=model_name, name=nom)
        epochs_list.append(epoch)
        loss_list.append(loss_metric.result())

plot_and_save_loss(loss_list, model_name)
