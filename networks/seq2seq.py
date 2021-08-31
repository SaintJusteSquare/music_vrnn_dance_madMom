import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import variable, zeros, concatenate
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRUCell, GRU

from networks.vrnn import Vrnn, Sampling


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


def inverse_sigmoid_decay(initial_value, global_step, decay_rate=1000.0,
                          name=None):
    """Applies inverse sigmoid decay to the decay variable (learning rate).
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies an inverse sigmoid decay
    function to a provided initial variable value.  It requires a `global_step`
    value to compute the decayed variable value. You can just pass a TensorFlow
    variable that you increment at each training step.
    The function returns the decayed variable value.  It is computed as:

    With decay-var = 1.0, gstep = x, decay_rate = 10000.0
    1.0*(10000.0/(10000.0+exp(x/(10000.0))))

    ```python
    decayed_var = decay_variable *
                  decay_rate / (decay_rate + exp(global_step / decay_rate))
    ```

    Rough Infos           | Value @ t=0 | (Real) decay start | Reaches Zero
    -------------------------------------------------------------------------
    decay_rate:    10.0   | 0.9         |          -40       |         100
    decay_rate:   100.0   | 0.985       |          -20       |       1,100
    decay_rate:  1000.0   | 1.0         |        2,000       |      12,000
    decay_rate: 10000.0   | 1.0         |       50,000       |     110,000

    Parameters
    ----------
    initial_value: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial variable value to decay.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate >> 1.
    name: String.  Optional name of the operation.  Defaults to
      'InvSigmoidDecay'
    Returns
    ----------
    A scalar `Tensor` of the same type as `decay_variable`.  The decayed
    variable value (such as learning rate).
    """
    assert decay_rate > 1, "The decay_rate has to be >> 1."

    initial_value = tf.convert_to_tensor(initial_value, name="initial_value")
    dtype = initial_value.dtype
    global_step = tf.cast(global_step, dtype)
    decay_rate = tf.cast(decay_rate, dtype)

    denom = decay_rate + tf.exp(global_step / decay_rate)
    return tf.multiply(initial_value, decay_rate / denom, name=name)


class Sampling(layers.Layer):
    """Uses (z_mean, z_std) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + z_std * epsilon

    def sample_sequence(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        timesteps = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, timesteps, dim))
        return z_mean + z_std * epsilon


class GRULayer(layers.Layer):

    def __init__(self, hdim, **kwargs):
        super().__init__(**kwargs)

        self.GRU1 = GRUCell(hdim, name='rnn1')
        self.GRU2 = GRUCell(hdim, name='rnn2')
        self.GRU3 = GRUCell(hdim, name='rnn3')
        self.GRU4 = GRUCell(hdim, name='rnn4')

    def call(self, inputs, states):
        [state1, state2, state3, state4] = states
        o1, [s1] = self.GRU1(inputs, [state1])
        o2, [s2] = self.GRU1(inputs, [state2])
        o3, [s3] = self.GRU1(inputs, [state3])
        outputs, [s4] = self.GRU2(o1, [state4])
        state_prime = [s1, s2, s3, s4]
        return outputs, state_prime

    def get_initial_state(self, inputs):
        h1 = self.GRU1.get_initial_state(inputs=inputs)
        h2 = self.GRU2.get_initial_state(inputs=inputs)
        h3 = self.GRU3.get_initial_state(inputs=inputs)
        return [h1, h2, h3]


class Seq2seq(tf.keras.Model):

    def __init__(self, model, x_dim, y_dim, y2s_dim, h_dim, z_dim, z2s_dim, q_z_dim, p_z_dim):
        super(Seq2seq, self).__init__()

        self.x_dim = x_dim
        self.x2s_dim = int(x_dim / 7)
        self.z_dim = z_dim
        self.target_dim = y_dim
        self.h_dim = h_dim

        self.base_model = model

        self.X_to_Z = Sequential(
            [
                Dense(self.x2s_dim, activation='relu', name='dense1'),
                GRU(self.x2s_dim, return_sequences=True, activation='relu', name='gru1'),
                GRU(self.x2s_dim, return_sequences=True, activation='relu', name='gru2'),
                GRU(self.x2s_dim, return_sequences=True, activation='relu', name='gru2'),
                GRU(self.z_dim, return_sequences=True, activation='relu', name='gru2'),
            ],
            name='X_to_Z'
        )

        # Feature extraction and transformation.
        self.Y_transform = Sequential(
            [
                Dense(y2s_dim, activation='relu', name='l1'),
                Dense(y2s_dim, activation='relu', name='l2'),
            ],
            name='Y_transform'
        )
        self.Z_transform = Sequential(
            [
                Dense(z2s_dim, activation='relu', name='l1'),
                Dense(z2s_dim, activation='relu', name='l2'),
            ],
            name='Z_transform'
        )

        # Recurrence
        # [x2s_dim + z2s_dim] -> h_dim + [h_dim]
        self.rnn = GRULayer(h_dim, name='gru_layer')

        # Decoder
        # [z2s_dim, h_dim] -> p_z_dim -> target_dim
        #                             -> target_dim
        self.theta = Sequential(
            [
                Dense(p_z_dim, activation='relu', name='l1'),
                Dense(p_z_dim, activation='relu', name='l2'),
            ],
            name='theta'
        )
        self.theta_mu = Dense(self.target_dim, activation=None, name='theta_mu')
        self.theta_sig = Dense(self.target_dim, activation='softplus', name='theta_sig')

        # Sampling
        self.sampling = Sampling()

    def transfert_weights(self):
        base_layers_name = ['X_transform',
                            'Z_transform',
                            'gru_layer',
                            'theta',
                            'theta_mu',
                            'theta_sig']

        host_layer_name = ['Y_transform',
                           'Z_transform',
                           'gru_layer',
                           'theta',
                           'theta_mu',
                           'theta_sig']

        for i in range(len(base_layers_name)):
            base_name = base_layers_name[i]
            host_name = host_layer_name[i]
            weights = self.base_model.get_layer(base_name).get_weights()
            layer = self.get_layer(host_name)
            layer.set_weights(weights)
            layer.trainable = False

    def call(self, inputs, training=None, mask=None, sample=False, scheduled_sampling_decay_rate=None, step=None):
        y_0, x = inputs
        if scheduled_sampling_decay_rate is None:
            if len(y_0.shape) != 2:
                raise ValueError('y_0 must be of 2 dims.')
        else:
            if len(y_0.shape) != 3:
                raise ValueError('y_0 must be of 3 dims.')
        batch = y_0.shape[0]
        if x.shape[0] != batch:
            raise ValueError('x_0 and z have incompatible batch dimension')
        timesteps = x.shape[1]

        reconstruction = list()
        theta_mu = list()
        theta_sig = list()

        H1 = variable(zeros((batch, self.h_dim)))
        H2 = variable(zeros((batch, self.h_dim)))
        H3 = variable(zeros((batch, self.h_dim)))
        H4 = variable(zeros((batch, self.h_dim)))
        state_t = [H1, H2, H3, H4]

        z = self.X_to_Z(x)
        zprime = self.Z_transform(z)
        yprime_t = self.Y_transform(y_0)

        if scheduled_sampling_decay_rate is not None:
            yprime_t = yprime_t[:, 0, :]

        if scheduled_sampling_decay_rate is None:

            for timestep in range(timesteps):

                zprime_t = zprime[:, timestep, :]
                _, state_t = self.rnn(concatenate([yprime_t, zprime_t]), states=state_t)

                theta_t = self.theta(concatenate([zprime_t, state_t[-1]]))
                theta_mu_t = self.theta_mu(theta_t)
                theta_sig_t = self.theta_sig(theta_t)

                reconstruction_t = self.sampling([theta_mu_t, theta_sig_t])

                yprime_t = self.Y_transform(reconstruction_t)
                reconstruction.append(reconstruction_t)
                theta_mu.append(theta_mu_t)
                theta_sig.append(theta_sig_t)

        else:

            is_training = True
            sampling_prob = inverse_sigmoid_decay(1.0, step, decay_rate=scheduled_sampling_decay_rate)
            uniform_random = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
            coin_success = tf.less(uniform_random, sampling_prob, name="coin_flip")
            # combine both decisions with logical switch, because nested tf.cond caused an error
            sample_from_gt = tf.math.logical_and(is_training, coin_success)

            for timestep in range(timesteps):

                zprime_t = zprime[:, timestep, :]
                _, state_t = self.rnn(concatenate([yprime_t, zprime_t]), states=state_t)

                theta_t = self.theta(concatenate([zprime_t, state_t[-1]]))
                theta_mu_t = self.theta_mu(theta_t)
                theta_sig_t = self.theta_sig(theta_t)

                reconstruction_t = self.sampling([theta_mu_t, theta_sig_t])

                motion_input = tf.cond(sample_from_gt,
                                       lambda: y_0[:, timestep, :],
                                       lambda: reconstruction_t,
                                       name="sample_switch")

                yprime_t = self.Y_transform(motion_input)
                reconstruction.append(motion_input)
                theta_mu.append(theta_mu_t)
                theta_sig.append(theta_sig_t)

        reconstruction = K.stack(reconstruction, axis=1)
        if sample:
            return reconstruction
        else:
            theta_mu = K.stack(theta_mu, axis=1)
            theta_sig = K.stack(theta_sig, axis=1)
            return reconstruction, theta_mu, theta_sig
