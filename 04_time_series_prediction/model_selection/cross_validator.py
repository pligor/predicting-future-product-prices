from __future__ import division

from sklearn.model_selection import StratifiedKFold, KFold
from interfaces.neural_net_model_interface import NeuralNetModelInterface
import numpy as np


class CrossValidator(NeuralNetModelInterface):
    which_set = 'train'

    def __init__(self, random_state, data_provider_class, stratified=True):  # batch_size,
        super(CrossValidator, self).__init__()
        self.__random_state = random_state
        # self.batch_size = batch_size
        self.__data_provider_class = data_provider_class
        self.__stratified = stratified

    def __getIterator(self, n_splits, batch_size, dp_with_batch_size=False, **data_provider_params):
        """
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        From documentation:
        Note that providing y is sufficient to generate the splits and hence np.zeros(n_samples) may be used as a
        placeholder for X instead of actual training data.
        """

        # batch_size=self.batch_size,
        if dp_with_batch_size:
            data_provider = self.__data_provider_class(which_set=self.which_set, batch_size=batch_size,
                                                       **data_provider_params)
        else:
            data_provider = self.__data_provider_class(which_set=self.which_set, **data_provider_params)

        data_len = data_provider.data_len

        assert (data_len / n_splits) % batch_size == 0, \
            "the splits should be chose in order to make the batch size fit: " \
            "input len: {}, n splits {}, input len / n splits {}, batch size {}".format(
                data_len, n_splits, data_len / n_splits, batch_size)

        KFoldClass = StratifiedKFold if self.__stratified else KFold

        kFold = KFoldClass(n_splits=n_splits, shuffle=True, random_state=self.__random_state)

        indices_iterator = kFold.split(np.zeros(data_len),  # this is allowed, check documentation for more info
                                       data_provider.targets) if self.__stratified else kFold.split(np.zeros(data_len))

        return indices_iterator, data_len

    def cross_validate(self, n_splits, batch_size, data_provider_params, **kwargs):
        results = []

        assert isinstance(data_provider_params, dict), "you should include data_provider_params in your parameters"

        if 'batch_size' in data_provider_params.keys():  # handling minor silly conflict
            assert data_provider_params['batch_size'] == batch_size
            indsIterator, inputLen = self.__getIterator(n_splits=n_splits, dp_with_batch_size=True,
                                                        **data_provider_params)
        else:
            indsIterator, inputLen = self.__getIterator(n_splits=n_splits, batch_size=batch_size,
                                                        **data_provider_params)

        for train_indices, valid_indices in indsIterator:
            train_data = self.__data_provider_class(which_set=self.which_set,
                                                    indices=train_indices,
                                                    **data_provider_params)

            valid_data = self.__data_provider_class(which_set=self.which_set,
                                                    indices=valid_indices,
                                                    **data_provider_params)

            if kwargs['preds_gather_enabled']:
                kwargs['preds_dp'] = self.__data_provider_class(which_set=self.which_set,
                                                                indices=valid_indices,
                                                                shuffle_order=False,
                                                                **data_provider_params)

            try:
                result = self.train_validate(train_data=train_data, valid_data=valid_data, **kwargs)
            except KeyError:
                result = self.train_validate(train_data=train_data, valid_data=valid_data, batch_size=batch_size,
                                             **kwargs)

            results.append(result)

        return results

    def train_validate(self, train_data, valid_data, **kwargs):
        raise NotImplementedError
