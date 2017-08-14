import os
import numpy as np
from numpy.lib.stride_tricks import as_strided

DEFAULT_SEED = 123456  # Default random number generator seed if none provided.

class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()


class OneOfKDataProvider(DataProvider):
    """1-of-K classification target data provider.

    Transforms integer target labels to binary 1-of-K encoded targets.

    Derived classes must set self.num_classes appropriately.
    """

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(OneOfKDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1-of-K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1-of-K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class MSDGenreDataProvider(OneOfKDataProvider):
    """Data provider for Million Song Dataset genre classification task."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, num_classes=10, window_size=120, stride=1, disturb_label=0.0):
        """Create a new Million Song Dataset genre data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the MSD genre data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'train-big', 'valid'], (
            'Expected which_set to be either train or valid. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        assert num_classes in [10, 25], (
            'Expected num_classes to be either 10 or 25. '
            'Got {0}'.format(num_classes)
        )
        self.num_classes = num_classes

        assert stride <= window_size, (
            'Expected stride to be less or equal then window_size'
        )
        self.window_size = window_size
        self.stride = stride

        self.disturb_label = disturb_label
        self.disturb_label_dist = np.eye(self.num_classes)*(1.-disturb_label)+disturb_label/self.num_classes

        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'],
            'msd-{0}-genre-{1}.npz'.format(num_classes,which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        # flatten inputs to vectors and upcast to float32
        inputs = inputs.reshape((inputs.shape[0], -1)).astype('float32')
        # label map gives strings corresponding to integer label targets
        self.label_map = loaded['label_map']
        # pass the loaded data to the parent class __init__
        super(MSDGenreDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    @property
    def inputs_shape(self):
        return (len(self.inputs),self.inputs.shape[1]//25//self.stride-self.window_size//self.stride+1,self.window_size*25)

    def create_sliding_windows(self, batch):
        _batch = batch.reshape((len(batch),-1,25))
        strides = list(_batch.strides)
        strides[1] *= self.stride
        shape = list(self.inputs_shape)
        shape[0] = len(_batch)
        return as_strided(_batch, shape, strides=strides)

    def next(self):
        inputs_batch, targets_batch = super(OneOfKDataProvider, self).next()

        if self.window_size == self.stride:
            splits = inputs_batch.shape[1]//(25*self.window_size)
            inputs_batch = inputs_batch.reshape((len(inputs_batch),splits,inputs_batch.shape[1]//splits))
        else:
            inputs_batch = self.create_sliding_windows(inputs_batch)

        if self.disturb_label > 0.0:
            targets_batch = np.copy(targets_batch)
            for i in range(len(targets_batch)):
                targets_batch[i] = self.rng.choice(np.arange(self.num_classes),p=self.disturb_label_dist[targets_batch[i]])

        if inputs_batch.shape[1] > 1:
            targets_batch = targets_batch.repeat(inputs_batch.shape[1],axis=0)

        return inputs_batch, self.to_one_of_k(targets_batch)
