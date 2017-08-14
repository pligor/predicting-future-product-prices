class NeuralNetModelInterface(object):
    def train_validate(self, train_data, valid_data, **kwargs):
        raise NotImplementedError
