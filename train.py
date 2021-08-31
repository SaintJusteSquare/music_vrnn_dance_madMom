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
from utils.dataset import Skeleton
from utils.plot_result import test_draw, draw_image, draw
from motion_transform import reverse_motion_transform

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
from networks.vrnn import Vrnn, Sampling
from networks.seq2seq import Seq2seq, Gaussian

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#############################
# Settings
#############################

BATCH_SIZE = 128
time_steps = 50

exp = 'exp1'
model_name = os.path.join(exp, 'variationalRNN')
model_gen_name = os.path.join(exp, 'music2dance_STFT_scheduled_sampling')

data_path = 'data'
configuration = {'file_pos_minmax': 'data/pos_minmax.h5',
                 'normalization': 'interval',
                 'rng_pos': [-0.9, 0.9]}
inference = True

if not os.path.isdir(exp):
    os.makedirs(exp)
if not os.path.isdir(model_name):
    os.makedirs(model_name)
if not os.path.isdir(model_gen_name):
    os.makedirs(model_gen_name)

################################
# Loading datasets (Train/Test)
################################

train_path = os.path.join(data_path, 'train')
train_generator = Skeleton(train_path, 'train', configuration, BATCH_SIZE, sequence=time_steps, init_step=1,
                           shuffle=True, set_type='float32')

test_path = os.path.join(data_path, 'test')
test_generator = Skeleton(test_path, 'test', configuration, BATCH_SIZE, sequence=time_steps, init_step=1,
                          shuffle=True, set_type='float32')

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

if not inference:
    epochs = 40

    x_dim = 69
    x2s_dim = 50
    z_dim = 50
    z2s_dim = 20

    k = 1
    h_dim = 1000

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150

    batch_size = BATCH_SIZE

    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50], [1e-4, 1e-5])
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_metric = tf.keras.metrics.Mean()

    model = Vrnn(x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim, q_z_dim=q_z_dim,
                 p_z_dim=p_z_dim, p_x_dim=p_x_dim, mode='gauss', k=k)
    sampler = Sampling()

    folder_images = os.path.join(exp, 'folder_image')
    if not os.path.isdir(folder_images):
        os.makedirs(folder_images)

    groundtruth_0 = train_generator.__getitem__(0)[1][0]
    first_frame = np.expand_dims(groundtruth_0[0], axis=0)
    groundtruth_0 = np.expand_dims(groundtruth_0, axis=0)

    test_theta_mu, test_theta_sig, test_z = model(groundtruth_0)
    reconstruction = sampler.sample_sequence([test_theta_mu, test_theta_sig])
    reconstruction = np.squeeze(reconstruction)
    reconstruction = reverse_motion_transform(reconstruction, configuration)
    nom = 'reconstruction_at_epoch_{:04d}.png'.format(0)
    draw_sequence(reconstruction, exp_folder=model_name, name=nom)

    fromDistrib = model.sample(first_frame, test_z)
    fromDistrib = np.array(fromDistrib)
    fromDistrib = np.squeeze(fromDistrib)
    fromDistrib = reverse_motion_transform(fromDistrib, configuration)
    nom = 'Sample_from_z_at_epoch{:04d}.png'.format(0)
    draw_sequence(fromDistrib, exp_folder=model_name, name=nom)

    groundtruth_0 = np.squeeze(groundtruth_0)
    groundtruth_0 = reverse_motion_transform(groundtruth_0, configuration)
    nom = 'groundtruth_sequence'
    draw_sequence(groundtruth_0, exp_folder=model_name, name=nom)

    epochs_list = list()
    loss_list = list()
    # val_loss_list = list()
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))
        start_time = time.time()
        for step in range(train_generator.__len__()):
            example_index = step * batch_size
            train_batch = train_generator.__getitem__(example_index)
            train_x = np.reshape(train_batch[1], (BATCH_SIZE, time_steps, x_dim))
            with tf.GradientTape() as tape:
                reconstructed = model(train_x)
                loss = model.losses

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer1.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
            if step == 300:
                break

        end_time = time.time()

        if epoch % 3 == 0:
            print('Epoch: {}, train: {}, ''time elapse for current epoch {}'.format(epoch, loss_metric.result(),
                                                                                    end_time - start_time))
            groundtruth_0 = train_generator.__getitem__(0)[1][0]
            first_frame = np.expand_dims(groundtruth_0[0], axis=0)
            groundtruth_0 = np.expand_dims(groundtruth_0, axis=0)

            test_theta_mu, test_theta_sig, test_z = model(groundtruth_0)
            reconstruction = sampler.sample_sequence([test_theta_mu, test_theta_sig])
            reconstruction = np.squeeze(reconstruction)
            reconstruction = reverse_motion_transform(reconstruction, configuration)
            nom = 'reconstruction_at_epoch_{:04d}.png'.format(epoch)
            draw_sequence(reconstruction, exp_folder=model_name, name=nom)

            fromDistrib = model.sample(first_frame, test_z)
            fromDistrib = np.squeeze(fromDistrib)
            fromDistrib = reverse_motion_transform(fromDistrib, configuration)
            nom = 'Sample_from_z_at_epoch{:04d}.png'.format(epoch)
            draw_sequence(fromDistrib, exp_folder=model_name, name=nom)

            groundtruth_0 = np.squeeze(groundtruth_0)
            groundtruth_0 = reverse_motion_transform(groundtruth_0, configuration)
            nom = 'groundtruth_sequence_at_epoch_{:04d}.png'.format(epoch)
            draw_sequence(groundtruth_0, exp_folder=model_name, name=nom)

    plot_and_save_loss(loss_list, model_name)
    checkpoint = os.path.join(model_name, 'mmodel_weights_at_epoch_'.format(epochs))
    model.save_weights(checkpoint)

else:
    #############################
    # Loading base model
    #############################
    epochs = 40
    x_dim = 69
    x2s_dim = 50
    z_dim = 50
    z2s_dim = 20

    k = 1
    h_dim = 1000

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150

    model = Vrnn(x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim, q_z_dim=q_z_dim,
                 p_z_dim=p_z_dim, p_x_dim=p_x_dim, mode='gauss', k=1)
    checkpoint = os.path.join(model_name, 'mmodel_weights_at_epoch_'.format(epochs))
    model.load_weights(checkpoint)
    model(train_generator.__getitem__(0)[1])
    print('model built? ', model.built)

    #############################
    # Models/training
    #############################

    epoch = 50
    SCHED_SAMPLING_DECAY = 1000.0

    x_dim = 1025 * 7
    y_dim = 69
    y2s_dim = 50
    z_dim = 50
    z2s_dim = 20
    h_dim = 1000

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150
    batch_size = BATCH_SIZE

    musicToDance = Seq2seq(model=model, x_dim=x_dim, y_dim=y_dim, y2s_dim=y2s_dim, h_dim=h_dim, z_dim=z_dim,
                           z2s_dim=z2s_dim, q_z_dim=q_z_dim, p_z_dim=p_z_dim)

    initial_learning_rate = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50], [1e-4, 1e-5])
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_metric = tf.keras.metrics.Mean()

    x_0, y_0 = train_generator.__getitem__(0)
    x_0 = np.reshape(x_0, (batch_size, time_steps, x_dim))
    first_frames = y_0[:, 0, :]

    reconstruction = musicToDance([first_frames, x_0], sample=True)
    reconstruction = reconstruction[0]
    reconstruction = np.array(reconstruction)
    reconstruction = reverse_motion_transform(reconstruction, configuration)
    nom = 'music2dance_at_epoch_{:04d}.png'.format(0)
    draw_sequence(reconstruction, exp_folder=model_gen_name, name=nom)

    musicToDance.transfert_weights()
    del model

    epochs_list = list()
    loss_list = list()
    # val_loss_list = list()
    global_step = 0
    for epoch in range(1, epochs + 1):
        print("Start of epoch %d" % (epoch,))
        start_time = time.time()
        for step in range(train_generator.__len__()):
            example_index = step * batch_size
            train_batch = train_generator.__getitem__(example_index)
            train_x = np.reshape(train_batch[0], (BATCH_SIZE, time_steps, x_dim))
            train_y = train_batch[1]
            # first_frames = train_y[:, 0, :]
            with tf.GradientTape() as tape:
                reconstruction, mu, sig = musicToDance.call([train_y, train_x],
                                                            scheduled_sampling_decay_rate=SCHED_SAMPLING_DECAY,
                                                            step=global_step)

                loss = tf.losses.mse(train_y, reconstruction) + Gaussian(train_y, mu, sig)

            grads = tape.gradient(loss, musicToDance.trainable_weights)
            optimizer1.apply_gradients(zip(grads, musicToDance.trainable_weights))

            global_step += 1

            loss_metric(loss)
            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
            if step == 500:
                break

        end_time = time.time()

        if epoch % 5 == 0:
            print('Epoch: {}, train: {}, ''time elapse for current epoch {}'.format(epoch, loss_metric.result(),
                                                                                    end_time - start_time))
            x_0, y_0 = train_generator.__getitem__(0)
            x_0 = np.reshape(x_0, (batch_size, time_steps, x_dim))
            first_frames = y_0[:, 0, :]
            reconstruction = musicToDance([first_frames, x_0], sample=True)
            reconstruction = reconstruction[0]
            reconstruction = np.array(reconstruction)
            reconstruction = reverse_motion_transform(reconstruction, configuration)
            nom = 'music2dance_at_epoch_{:04d}.png'.format(epoch)
            draw_sequence(reconstruction, exp_folder=model_gen_name, name=nom)
            groundtruth = y_0[0]
            groundtruth = reverse_motion_transform(groundtruth, configuration)
            nom = 'groundtruth_sequence_at_epoch_{:04d}.png'.format(epoch)
            draw_sequence(groundtruth, exp_folder=model_gen_name, name=nom)
            epochs_list.append(epoch)
            loss_list.append(loss_metric.result())

    plot_and_save_loss(loss_list, model_gen_name)
    print("Done !")
