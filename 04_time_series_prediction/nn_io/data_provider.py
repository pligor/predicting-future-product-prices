# -*- coding: utf-8 -*-
import numpy as np
from nn_io import DEFAULT_SEED
from interfaces.iterable import Iterable


class UnifiedDataProvider(Iterable):
    """Generic data provider."""
    DEFAULT_FIRST_EPOCH = 0

    def __init__(self, datalist, batch_size,
                 max_num_batches=-1, shuffle_order=True, rng=None, initial_order=None):
        """Create a new data provider object.

        Args:
            data list of (ndarray): Array of shape (num_data, some_dim).
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        assert len(datalist) >= 2, "input and targets should be the first two items and then whatever else you want"

        self.datalist = datalist
        # self.inputs = datalist[0]
        # self.targets = datalist[1]

        data_len = datalist[0].shape[0]
        self.data_len = data_len

        assert np.all([data.shape[0] == data_len for data in datalist]), \
            "all parts of data list should have the same length"

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order

        assert initial_order is None or (len(initial_order) == data_len and len(initial_order.shape) == 1)
        self._initial_order = np.arange(data_len) if initial_order is None else initial_order
        self._current_order = self._initial_order

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        self._curr_batch = self.DEFAULT_FIRST_EPOCH
        self.new_epoch()

    @property
    def current_order(self):
        return self._current_order.copy()

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
        possible_num_batches = self.data_len // self.batch_size
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
        self._curr_batch = self.DEFAULT_FIRST_EPOCH
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        # inv_perm = np.argsort(self. current order)
        # more generic case:
        inv_perm = np.argwhere(self._current_order == self._initial_order[np.newaxis].T)[:, 1]

        self._current_order = self._current_order[inv_perm]

        self.datalist = [data[inv_perm] for data in self.datalist]

        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.data_len)
        self._current_order = self._current_order[perm]

        self.datalist = [data[perm] for data in self.datalist]

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

        batch_list = [data[batch_slice] for data in self.datalist]

        self._curr_batch += 1

        return batch_list

    # Python 3.x compatibility
    def __next__(self):
        return self.next()


class DataProvider(Iterable):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None, initial_order=None):
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

        assert initial_order is None or (len(initial_order) == inputs.shape[0] and len(initial_order.shape) == 1)
        self._initial_order = np.arange(inputs.shape[0]) if initial_order is None else initial_order
        self._current_order = self._initial_order

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
        # inv_perm = np.argsort(self. current order)
        # more generic case:
        inv_perm = np.argwhere(self._current_order == self._initial_order[np.newaxis].T)[:, 1]

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

class SubSampleableDataProvider(object):
    """This class is helpful in cases you have many samples and you want to process less, typically due to computational
    restrictions on your hardware resources"""

    @staticmethod
    def subsample_dic(dic, target_size_or_fraction=None, random_state=None):
        """dic is a dictionary that contains arrays
        typical usage with np.load from an npz file
        All arrays contained in dic must be of same length
        If target_size_or_fraction is lower than one than one then we calculate a fraction of the length, otherwise it is an absolute number"""

        if target_size_or_fraction is None:
            return dic
        else:
            assert target_size_or_fraction > 0.

            random_state = np.random if random_state is None else random_state

            keys = dic.keys()
            assert len(keys) > 0, "dic cannot be empty"
            cur_len = len(dic[keys[0]])

            assert target_size_or_fraction <= cur_len, "target_size cannot be larger than the available length"

            for key in keys:
                assert cur_len == len(dic[key]), "All arrays contained in dic must be of same length"

            target_size = int(
                cur_len * target_size_or_fraction) if target_size_or_fraction < 1. else target_size_or_fraction

            random_inds = random_state.choice(cur_len, target_size, replace=False)

            subsampled_dic = {}
            for key in keys:
                subsampled_dic[key] = dic[key][random_inds]

            return subsampled_dic


class FilterableDataProvider(UnifiedDataProvider):
    def __init__(self, datalist, batch_size,
                 max_num_batches=-1, shuffle_order=True, rng=None,
                 data_filtering=None, initial_order=None):
        # just giving the ability to set None
        data_filtering = (lambda identity: identity) if data_filtering is None else data_filtering

        # print type(data_filtering)
        # print type(data_filtering(inputs, targets))
        filtered_datalist = [data_filtering(cur_data) for cur_data in datalist]

        super(FilterableDataProvider, self).__init__(datalist=filtered_datalist,
                                                     batch_size=batch_size,
                                                     max_num_batches=max_num_batches,
                                                     shuffle_order=shuffle_order,
                                                     rng=rng,
                                                     initial_order=initial_order)


class CrossValDataProvider(FilterableDataProvider):
    def __init__(self, datalist, batch_size, max_num_batches=-1, shuffle_order=True, rng=None,
                 indices=None):
        indices = indices if indices is None else np.array(indices)  # ensuring it is a numpy array if necessary

        data_filtering = indices if indices is None else (lambda cur_data: cur_data[indices])

        super(CrossValDataProvider, self).__init__(datalist=datalist, batch_size=batch_size,
                                                   max_num_batches=max_num_batches,
                                                   shuffle_order=shuffle_order, rng=rng,
                                                   data_filtering=data_filtering, initial_order=indices)


class RnnStaticLenDataProvider(Iterable):
    """."""

    def _RnnStaticLen_init(self, batch_size, truncated_backprop_len, series_total_fixed_len,
                           dimensionality_of_point_in_time,
                           process_targets=False):
        """segment count is into how many pieces does the inputs are splitted because of the truncated backprop length"""

        self.dimensionality_of_point_in_time = dimensionality_of_point_in_time
        self.__batch_size = batch_size

        segment_part_count = series_total_fixed_len / truncated_backprop_len
        assert segment_part_count == int(segment_part_count)
        self.segment_part_count = int(segment_part_count)
        self.truncated_backprop_len = truncated_backprop_len

        self.__counter = 0
        self.__inputs_batch = None
        self.__targets_batch = None

        self.__targets_processor = process_targets if callable(process_targets) else self.__reshape
        self.__process_targets = True if callable(process_targets) else process_targets

    def __reshape(self, batch):
        return batch.reshape(
            (self.__batch_size,
             -1,  # self.segment_part_count,
             self.truncated_backprop_len, self.dimensionality_of_point_in_time)
        )

    def next(self):
        """."""
        if self.__counter % self.segment_part_count == 0:
            inputs_batch, targets_batch = super(RnnStaticLenDataProvider, self).next()

            self.__inputs_batch = self.__reshape(batch=inputs_batch)
            self.__counter = 0
            self.__targets_batch = self.__targets_processor(targets_batch) if self.__process_targets else targets_batch

        cur_input_batch = self.__inputs_batch[:, self.__counter, :, :]

        cur_target_batch = self.__targets_batch[:, self.__counter, :, :] \
            if self.__process_targets else self.__targets_batch

        cur_counter = self.__counter

        self.__counter += 1

        return (cur_input_batch, cur_target_batch), cur_counter


class OneOfKDataProvider(Iterable):  # DataProvider
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
