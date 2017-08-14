from sklearn.model_selection import StratifiedKFold
from interfaces.neural_net_model_interface import NeuralNetModelInterface


class CrossValidator(NeuralNetModelInterface):
    which_set = 'train'

    def __init__(self, random_state, data_provider_class):  # batch_size,
        super(CrossValidator, self).__init__()
        self.random_state = random_state
        # self.batch_size = batch_size
        self.data_provider_class = data_provider_class

    def __getIterator(self, n_splits, batch_size, **data_provider_params):
        """
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        From documentation:
        Note that providing y is sufficient to generate the splits and hence np.zeros(n_samples) may be used as a
        placeholder for X instead of actual training data.
        """
        kFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        data_provider = self.data_provider_class(which_set=self.which_set,  # batch_size=self.batch_size,
                                                 **data_provider_params)

        inputLen = len(data_provider.inputs)

        assert (len(data_provider.inputs) / n_splits) % batch_size == 0, \
            "the splits should be chose in order to make the batch size fit"

        return kFold.split(data_provider.inputs, data_provider.targets), inputLen

    def cross_validate(self, n_splits, batch_size, data_provider_params, **kwargs):
        stats_list = []

        assert isinstance(data_provider_params, dict), "you should include data_provider_params in your parameters"

        indsIterator, inputLen = self.__getIterator(n_splits=n_splits, batch_size=batch_size, **data_provider_params)

        for train_indices, valid_indices in indsIterator:
            train_data = self.data_provider_class(which_set=self.which_set,
                                                  # batch_size=self.batch_size,
                                                  indices=train_indices,
                                                  **data_provider_params)

            valid_data = self.data_provider_class(which_set=self.which_set,
                                                  # batch_size=self.batch_size,
                                                  indices=valid_indices,
                                                  **data_provider_params)

            stats = self.train_validate(train_data=train_data, valid_data=valid_data, **kwargs)

            stats_list.append(stats)

        return stats_list

    def train_validate(self, train_data, valid_data, **kwargs):
        raise NotImplementedError
