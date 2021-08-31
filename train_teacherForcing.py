#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import plot_model
from tensorflow import convert_to_tensor
import tensorflow.keras.backend as K

# tf.logging.set_verbosity(tf.logging.DEBUG)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys

module_utils = os.path.join(os.getcwd(), 'utils')
sys.path.append(module_utils)
from utils.dataset import Skeleton
from utils.plot_result import test_draw, draw_image, draw
from motion_transform import reverse_motion_transform

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#############################
# Settings
#############################

BATCH_SIZE = 128
time_steps = 51

exp = 'exp_teacherForcing'
model_name = os.path.join(exp, 'seqseq')

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
# model functions
#############################

def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = 0.5 * K.sum(K.square(y - mu) / sig ** 2 + 2 * K.log(sig) + K.log(2 * np.pi), axis=-1)
    return K.mean(nll)


def sample_sequence(inputs):
    z_mean, z_std = inputs
    batch = tf.shape(z_mean)[0]
    timesteps = tf.shape(z_mean)[1]
    dim = tf.shape(z_mean)[2]
    epsilon = tf.keras.backend.random_normal(shape=(batch, timesteps, dim))
    return z_mean + z_std * epsilon


#############################
# Models/training 1
#############################

encoder_dim = 128 * 7
latent_dim = 128
decoder_dim = 69

encoder_inputs = Input(shape=(None, encoder_dim))
encoder1 = LSTM(latent_dim, activation='relu', return_sequences=True)(encoder_inputs)
encoder2 = LSTM(latent_dim, activation='relu', return_sequences=True)(encoder1)
encoder3 = LSTM(latent_dim, activation='relu', return_sequences=True)(encoder2)
encoder = LSTM(latent_dim, activation='relu', return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder3)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, decoder_dim))
decoder_lstm1 = LSTM(latent_dim, activation='relu', return_sequences=True)
decoder_lstm2 = LSTM(latent_dim, activation='relu', return_sequences=True)
decoder_lstm3 = LSTM(latent_dim, activation='relu', return_sequences=True)
decoder_lstm4 = LSTM(latent_dim, activation='relu',  return_sequences=True, return_state=True)
decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
decoder_outputs = decoder_lstm2(decoder_outputs)
decoder_outputs = decoder_lstm3(decoder_outputs)
decoder_outputs, _, _ = decoder_lstm4(decoder_outputs)
decoder_dense = Dense(latent_dim, activation='relu')
decoder_mu = Dense(decoder_dim, activation=None)
decoder_sig = Dense(decoder_dim, activation='softplus')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_outputs_mu = decoder_mu(decoder_outputs)
decoder_outputs_sig = decoder_sig(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], [decoder_outputs_mu, decoder_outputs_sig])
plot_model(model, to_file=os.path.join(model_name, 'model.png'))

X, y = train_generator.__getitem__(0)
X = np.reshape(X[0], (1, 101, encoder_dim))
input_y = np.expand_dims(y[0][:-1], axis=0)
groundtruth = np.expand_dims(y[0][1:], axis=0)

reconstruction_mu, reconstruction_sig = model.predict([X, input_y])
reconstruction = sample_sequence([reconstruction_mu, reconstruction_sig])
reconstruction = np.squeeze(reconstruction)
reconstruction = reverse_motion_transform(reconstruction, configuration)
nom = 'reconstruction_at_epoch_{:04d}.png'.format(0)
draw_sequence(reconstruction, exp_folder=model_name, name=nom)

decoder_input = np.squeeze(groundtruth)
decoder_input = reverse_motion_transform(decoder_input, configuration)
nom = 'groundtruth_sequence'
draw_sequence(decoder_input, exp_folder=model_name, name=nom)

epochs = 100

initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50], [1e-3, 1e-4])
optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_metric = tf.keras.metrics.Mean()

epochs_list = list()
loss_list = list()
# val_loss_list = list()
for epoch in range(1, epochs + 1):
    print("Start of epoch %d" % (epoch,))
    start_time = time.time()
    for step in range(train_generator.__len__()):
        example_index = step * BATCH_SIZE
        encoder_input_batch, decoder_batch = train_generator.__getitem__(example_index)
        encoder_input_batch = convert_to_tensor(np.reshape(encoder_input_batch, (BATCH_SIZE, time_steps, encoder_dim)))
        decoder_input_batch = convert_to_tensor(decoder_batch[:, :-1, :])
        groundtruth_batch = convert_to_tensor(decoder_batch[:, 1:, :])
        with tf.GradientTape() as tape:
            mu, sig = model([encoder_input_batch, decoder_input_batch], training=True)
            reconstruction = sample_sequence([mu, sig])
            gaussian_loss = Gaussian(groundtruth_batch, mu, sig)
            if np.isnan(np.array(gaussian_loss)):
                loss = tf.losses.mse(groundtruth_batch, reconstruction)
            else:
                loss = Gaussian(groundtruth_batch, mu, sig) + tf.losses.mse(groundtruth_batch, reconstruction)

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

#############################
# Models/training 2
#############################

encoder_model = Model(encoder_inputs, encoder_states)
plot_model(encoder_model, to_file=os.path.join(model_name, 'encoder_model.png'))

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs)
decoder_outputs = decoder_lstm2(decoder_outputs)
decoder_outputs = decoder_lstm3(decoder_outputs)
decoder_outputs, state_h, state_c = decoder_lstm4(decoder_outputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_outputs_mu = decoder_mu(decoder_outputs)
decoder_outputs_sig = decoder_sig(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs_mu, decoder_outputs_sig] + decoder_states)
plot_model(decoder_model, to_file=os.path.join(model_name, 'decoder_model.png'))


def decode_sequence(input_seq, first_frame):
    nb_frames = input_seq.shape[1]
    nb_seq = int(nb_frames/time_steps)
    remaining_seq = nb_frames - time_steps * nb_seq

    target_frame = first_frame
    decoded_sequence = [first_frame]
    for i in range(nb_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq[i*time_steps:(i+1)*time_steps])

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False

        while not stop_condition:
            output_mu, output_sig, h, c = decoder_model.predict([target_frame] + states_value)
            output = sample_sequence([output_mu, output_sig])
            frame = np.squeeze(output)
            decoded_sequence.append(frame)

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sequence) >= nb_seq*(i+1):
                stop_condition = True

            target_frame = output

            # Update states
            states_value = [h, c]

    if remaining_seq > 0:
        states_value = encoder_model.predict(input_seq[nb_seq*time_steps:])
        stop_condition = False

        while not stop_condition:
            output_mu, output_sig, h, c = decoder_model.predict([target_frame] + states_value)
            output = sample_sequence([output_mu, output_sig])
            frame = np.squeeze(output)
            decoded_sequence.append(frame)

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sequence) >= nb_frames:
                stop_condition = True

            target_frame = output

            # Update states
            states_value = [h, c]

    decoded_sequence = np.stack(decoded_sequence, axis=0)
    return decoded_sequence


set_type = 'float32'

item = './data/train/trainf_000.h5'
file_name, input_data, motion_data, position_config = train_generator.get_file_data(item, type=set_type)
input_data = np.reshape(input_data, (1, input_data.shape[0], encoder_dim))
first_frame = np.expand_dims(motion_data[0, :], axis=0)
groundtruth = reverse_motion_transform(motion_data, configuration)
reconstruction = decode_sequence(input_data, first_frame)
print('reconstruction shape = ', reconstruction.shape)
draw_sequence(reconstruction, exp_folder=model_name, name='reconstruction_' + file_name)
draw_sequence(groundtruth, exp_folder=model_name, name=file_name)

item = './data/train/trainf_027.h5'
file_name, input_data, motion_data, position_config = train_generator.get_file_data(item, type=set_type)
input_data = np.reshape(input_data, (1, input_data.shape[0], encoder_dim))
first_frame = np.expand_dims(motion_data[0, :], axis=0)
groundtruth = reverse_motion_transform(motion_data, configuration)
reconstruction = decode_sequence(input_data, first_frame)
draw_sequence(reconstruction, exp_folder=model_name, name='reconstruction_' + file_name)
draw_sequence(groundtruth, exp_folder=model_name, name=file_name)
