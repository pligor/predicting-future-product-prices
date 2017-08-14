import numpy as np
from os import path
from collections import OrderedDict
from arima_estimator import ArimaEstimator


class ArimaCV(object):
    @staticmethod
    def cross_validate(inputs, targets, cartesian_combinations, score_dic_filepath=None, easy_mode=True):
        if score_dic_filepath is None:
            scoredic = OrderedDict()  # just empty dic
        else:
            scoredic = np.load(score_dic_filepath)[()] if path.isfile(score_dic_filepath) else OrderedDict()

        for pp, dd, qq in cartesian_combinations:
            cur_tuple = (pp, dd, qq)

            if cur_tuple in scoredic:
                continue
            else:
                ae = ArimaEstimator(p_auto_regression_order=pp, d_integration_level=dd, q_moving_average=qq,
                                    easy_mode=easy_mode)
                result = ae.fit(inputs=inputs, targets=targets).score(inputs=inputs, targets=targets)

                scoredic[cur_tuple] = result

                if score_dic_filepath is not None:
                    np.save(score_dic_filepath, scoredic)  # save on every iteration to be sure

        return scoredic
