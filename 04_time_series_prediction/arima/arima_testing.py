import numpy as np
import warnings

from arima.arima_estimator import ArimaEstimator


class ArimaTesting(object):
    @staticmethod
    def full_testing(npz_full, best_params, target_len):
        dic = np.load(npz_full)

        scores = []
        preds_collection = []
        keys = []

        for ii, cur_key in enumerate(dic):
            print ii, ",", cur_key
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                cur_sku = dic[cur_key][()]

                test_mat = cur_sku['test']

                test_ins = test_mat[:-target_len]
                test_ins_vals = test_ins.values.reshape(1, -1)

                test_tars = test_mat[-target_len:]
                test_tars_vals = test_tars.values.reshape(1, -1)

                ae = ArimaEstimator(p_auto_regression_order=best_params[0],
                                    d_integration_level=best_params[1],
                                    q_moving_average=best_params[2],
                                    easy_mode=False)
                cur_score = ae.fit(test_ins_vals, test_tars_vals).score(test_ins_vals, test_tars_vals)
                scores.append(cur_score)

                preds_collection.append(ae.preds.flatten())

            keys.append(cur_key)


        return keys, scores, np.array(preds_collection)
