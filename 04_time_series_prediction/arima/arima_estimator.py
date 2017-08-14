from numpy.linalg import LinAlgError
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from fastdtw import fastdtw
from time import sleep


class ArimaEstimator(object):
    def __init__(self, p_auto_regression_order, d_integration_level, q_moving_average, easy_mode,
                 method='css',
                 # http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.fit.html#statsmodels.tsa.arima_model.ARIMA.fit
                 ):
        super(ArimaEstimator, self).__init__()
        self.p_auto_regression_order = p_auto_regression_order
        self.d_integration_level = d_integration_level
        self.q_moving_average = q_moving_average
        self.easy_mode = easy_mode
        self.target_len = None
        self.dtw_score = None
        self.preds = None
        self.method = method

    def fit(self, inputs, targets):
        input_len = len(inputs)

        assert input_len == len(targets) and len(targets) > 0
        self.target_len = len(targets[0])

        dtw_scores = []
        preds = []
        for input_stream, target_stream in zip(inputs, targets):
            # print input_stream.shape
            # print target_stream.shape
            pred_stream = self.__get_forecast(input_stream=input_stream, target_stream=target_stream)

            if pred_stream is None:
                pass  # do not append anything
            else:
                assert len(pred_stream) == self.target_len
                cur_dtw_score = fastdtw(target_stream, pred_stream)[0]
                dtw_scores.append(cur_dtw_score)
                # print pred_stream.shape
                preds.append(pred_stream.flatten())
                # print

        self.preds = np.array(preds)

        if len(dtw_scores) == 0:
            self.dtw_score = np.NaN
        else:
            self.dtw_score = np.mean(dtw_scores)

        return self

    def score(self, inputs, targets):
        return self.dtw_score

    def __get_forecast(self, input_stream, target_stream):
        target_stream = list(target_stream)
        history = list(input_stream)
        pred_stream = []

        # start_params = [0.01] * np.sum(self.params)

        for cur_target in target_stream:
            # sleep(1)
            try:
                forecasted_value = \
                    ARIMA(history, order=self.params).fit(method=self.method, disp=0, transparams=False).forecast()[0]
            except LinAlgError:
                # print "this is the history: {}".format(history)
                forecasted_value = history[-1]  # just take the dummy prediction of predicting the latest value
            except ValueError:
                return None

            cur_pred = forecasted_value  # 0th is the index of the prediction

            if self.easy_mode:
                history.append(cur_target)
            else:
                history.append(cur_pred)

            pred_stream.append(cur_pred)

            # print "predicted: {}, expected: {}".format(y_hat, observation)

        return np.array(pred_stream)

    @property
    def params(self):
        return self.p_auto_regression_order, self.d_integration_level, self.q_moving_average
