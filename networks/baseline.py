import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import variable, zeros, concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, LSTMCell


class CNNFeat(tf.keras.Model):

    def __init__(self, dim):
        super(CNNFeat, self).__init__()

        self.conv1 = Conv2D()
        self.cvbn1 = BatchNormalization()
        self.conv2 = Conv2D()
        self.cvbn2 = BatchNormalization()
        self.conv3 = Conv2D()
        self.cvbn3 = BatchNormalization()
        self.conv4 = Conv2D()
        self.cvbn4 = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        cv1 = self.conv1(inputs)
        cvbn1 = self.cvbn1(cv1)
        cv2 = self.conv2(cvbn1)
        cvbn2 = self.cvbn1(cv2)
        cv3 = self.conv2(cvbn2)
        cvbn3 = self.cvbn1(cv3)
        cv4 = self.conv2(cvbn3)
        cvbn4 = self.cvbn1(cv4)
        return cvbn4


class Dancer(tf.keras.Model):

    def __init__(self, units, dim, out):
        super(Dancer, self).__init__()

        self.audiofeat = CNNFeat(dim)
        self.enc_lstm1 = LSTMCell(units)
        self.enc_lstm2 = LSTMCell(units)
        self.enc_lstm3 = LSTMCell(units)
        self.fc01 = LSTMCell(dim)
        self.dec_lstm1 = LSTMCell(units)
        self.dec_lstm2 = LSTMCell(units)
        self.dec_lstm3 = LSTMCell(units)
        self.out_signal = LSTMCell(out)

        self.state = {'ec1': None, 'ec2': None, 'ec3': None,
                      'eh1': None, 'eh2': None, 'eh3': None,
                      'dc1': None, 'dc2': None, 'dc3': None,
                      'dh1': None, 'dh2': None, 'dh3': None}
        self.delta_sgn = 0
        self.ot = 0
        self.h = 0

    def __call__(self, variables):
        xp = self.xp
        in_audio, curr_step, nx_step = variables
        batchsize, sequence = in_audio.shape[0:2]
        self.loss = loss_mse = loss_ctte = 0
        state = self.state
        self.ot = xp.std(curr_step, axis=1)
        for i in range(sequence):
            h, state, y = self.forward(state, curr_step, self.audiofeat(in_audio[:, i]), True)
            loss_mse += F.mean_squared_error(nx_step[:, i], y)
            curr_step = y
            ot = xp.std(nx_step[:, i], axis=1) * batchsize  # y
            delta_sgn = xp.sign(ot - self.ot)
            if i > 0:
                labels = xp.minimum(xp.absolute(delta_sgn + self.delta_sgn), 1)
                labels = xp.asarray(labels, dtype=xp.int32)
                loss2 = F.contrastive(h, self.h, labels, margin=3.0) / sequence
                # .mean_squared_error mean_absolute_error
                if float(chainer.cuda.to_cpu(loss2.data)) > 0.:
                    loss_ctte += loss2  # F.mean_squared_error mean_absolute_error
            self.h = h
            self.ot = ot
            self.delta_sgn = delta_sgn
        loss = loss_mse + loss_ctte
        self.loss = loss
        chainer.report({'loss': loss,
                        'loss_mse': loss_mse,
                        'loss_ctte': loss_ctte
                        }, self)
        stdout.write('loss={:.04f}\r'.format(float(chainer.cuda.to_cpu(loss.data))))
        stdout.flush()
        return self.loss

    def forward(self, state, h1, h, eval=False):
        act = F.elu
        ec1, eh1 = self.enc_lstm1(state['ec1'], state['eh1'], h)
        ec2, eh2 = self.enc_lstm2(state['ec2'], state['eh2'], eh1)
        ec3, eh3 = self.enc_lstm3(state['ec3'], state['eh3'], eh2)
        _h = F.tanh(self.fc01(eh3))
        h = F.concat((h1, _h))
        dc1, dh1 = self.dec_lstm1(state['dc1'], state['dh1'], h)
        dc2, dh2 = self.dec_lstm2(state['dc2'], state['dh2'], dh1)
        dc3, dh3 = self.dec_lstm3(state['dc3'], state['dh3'], dh2)
        h = act(self.out_signal(dh3))
        new_state = dict()
        for key in state:
            new_state[key] = locals()[key]
        if eval:
            return _h, new_state, h
        return new_state, h

