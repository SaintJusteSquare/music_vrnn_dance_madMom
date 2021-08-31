from abc import ABC
import os

import h5py
import glob
import logging
import keras
import numpy as np

from motion_transform import reverse_motion_transform


class Skeleton(keras.utils.Sequence, ABC):

    def __init__(self, folder, stage, configuration, batch=64, sequence=None, init_step=None, shuffle=True,
                 set_type=None):
        self._inputs = ["input", "motion"]
        self.n_channels = 1
        self.batch = batch
        self.steps = 1
        self.shuffle = shuffle
        self.fn = '{}/{}*'.format(folder, stage)
        self.list_file = glob.glob('{}/{}*'.format(folder, stage))
        self.configuration = configuration
        self.sequence = sequence
        self.init_step = init_step
        self._type = None
        self.idxs = None
        self.set_type = set_type

        # config of the processing of the dataset:
        self.norm = configuration['normalization']

        self.nb_files()
        if sequence is not None:
            self._arrange_sequence(sequence, init_step)

    def nb_files(self):
        n = len(self.list_file)
        print('In {}, {} files were founds'.format(self.fn, n))

    def _arrange_sequence(self, sequence, init_step):
        index = []
        for i in range(len(self.list_file)):
            with h5py.File(self.list_file[i], 'r') as f:
                current_lenght = f[self._inputs[0]].shape[0]
                if self.sequence >= current_lenght:
                    logging.error('The lenght of the sequence is larger thant the lenght of the file...')
                    raise ValueError('')
                max_size = current_lenght - (self.sequence + self.steps)
                if not '_dims' in locals():
                    _dims = [None] * len(self._inputs)
                    _types = [None] * len(self._inputs)
                    for j in range(len(self._inputs)):
                        testfile = f[self._inputs[j]]
                        _dim = testfile[0].shape
                        _dims[j] = _dim if len(_dim) != 0 else []
                        _types[j] = testfile.dtype
                        logging.info('  data label: {} \t dim: {} \t dtype: {}'.format(self._inputs[j], list(_dims[j]),
                                                                                       _types[i]))
            _index = [[i, x] for x in np.arange(max_size)]
            index += _index
        try:
            self._dims = _dims
        except Exception as e:
            logging.error('Cannot assign dimensions, data not found...')
            raise TypeError(e)
        self._type = _types
        self.idxs = index
        self.init_step = init_step
        self.on_epoch_end()
        logging.info('sequence: {}'.format(self.sequence))
        logging.info('Total of {} files...'.format(len(self.idxs)))

    def __len__(self):
        return int(np.floor(len(self.idxs) / self.batch))

    def get_idx(selfs):
        return selfs.idxs

    def get_type(self):
        return self._type

    def get_dim(self):
        return self._dims

    def get_configuration(self):
        return self.configuration

    def get_example(self, i):
        # TODO: To change according the train model
        # Currently, fits the train_lstm.
        iDB, iFL = self.idxs[i]
        data_labels = [None] * 2
        with h5py.File(self.list_file[iDB], 'r') as f:
            data_labels[0] = f[self._inputs[0]][iFL: iFL + self.sequence][None, :]
            data_labels[1] = f[self._inputs[1]][iFL: iFL + self.sequence][None, :]
        return data_labels

    def get_dataset(self, type=None, untransformed=False):
        """
        :param type: One who wishes to change type of the data has to precise it (str).
        :param untransformed: If True, return unormalized data.
        :return: dataset in the numpy format of shape (nb_examples, sequence, features)
        """
        # TODO: unfinished
        if self.sequence is not None:
            l = len(self.get_idx())
            X = np.zeros((l, self.sequence, self.get_dim()[0][0], self.get_dim()[0][1]))
            y = np.zeros((l, self.sequence, self.get_dim()[1][0]))
            if untransformed:
                for i in range(l):
                    motion_seq = reverse_motion_transform(np.squeeze(self.get_example(i)[0]), self.configuration)
                    X[i] = motion_seq
            else:
                for i in range(l):
                    X[i] = np.squeeze(self.get_example(i)[0])
            if type is not None:
                return X.astype(type)
            return X
        else:
            X = None
            for i in range(len(self.list_file)):
                with h5py.File(self.list_file[i], 'r') as f:
                    if not '_dims' in locals():
                        testfile = f[self._inputs]
                        _dims = testfile[0].shape
                        _types = testfile.dtype
                        logging.info(
                            '  data label: {} \t dim: {} \t dtype: {}'.format(self._inputs, list(_dims), _types))

                        X = np.array(f[self._inputs])
                    else:
                        x_temp = np.array(f[self._inputs])
                        X = np.concatenate([X, x_temp])
            return X

    def get_file_data(self, item, type=None, untransformed=False):
        # item = './data/train/trainf000.h5'
        with h5py.File(item, 'r') as f:
            name = str(np.array(f['sound_path']))[1:]
            input_data = np.array(f['input'])
            motion_data = np.array(f['motion'])
            config = np.array(f['position'])
        if untransformed:
            motion_data = reverse_motion_transform(motion_data, self.configuration)
        if type is not None:
            input_data = input_data.astype(type)
            motion_data = motion_data.astype(type)
        return name, input_data, motion_data, config

    def __getitem__(self, index):
        X_seq = np.empty((self.batch, self.sequence, self._dims[0][0], self._dims[0][1]))
        y_seq = np.empty((self.batch, self.sequence, self._dims[1][0]))
        for i in range(index, index + self.batch):
            example = self.get_example(i)
            t = i - index
            X_seq[t] = example[0]
            y_seq[t] = example[1]
        if self.sequence == 1:
            X_seq = np.squeeze(X_seq)
            y_seq = np.squeeze(y_seq)
        if self.set_type is not None:
            X_seq = X_seq.astype(self.set_type)
            y_seq = y_seq.astype(self.set_type)
        return X_seq, y_seq

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)


if __name__ == '__main__':
    # TODO: test.

    BATCH_SIZE = 128
    time_steps = 101

    data_path = '../data'
    configuration = {'file_pos_minmax': '../data/pos_minmax.h5',
                     'normalization': 'interval',
                     'rng_pos': [-0.9, 0.9]}

    train_path = os.path.join(data_path, 'train')
    train_generator = Skeleton(train_path, 'train', configuration, BATCH_SIZE, sequence=time_steps, init_step=1,
                               shuffle=True, set_type='float32')

    item = '../data/train/trainf_000.h5'
    set_type = 'float32'

    file_name, input_data, motion_data, position_config = train_generator.get_file_data(item, type=set_type)
    print(file_name)
    print('input shape = ', input_data.shape)
    print('motion shape = ', motion_data.shape)
    print('config: ', position_config)

    item = '../data/train/trainf_027.h5'
    set_type = 'float32'

    file_name, input_data, motion_data, position_config = train_generator.get_file_data(item, type=set_type)
    print(file_name)
    print('input shape = ', input_data.shape)
    print('motion shape = ', motion_data.shape)
    print('config: ', position_config)