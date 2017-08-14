import numpy as np


class PriceHistoryPack(object):
    def __init__(self, input_seq_len, num_features, target_seq_len):
        super(PriceHistoryPack, self).__init__()
        self.sku_ids = []
        self.XX = np.empty((0, input_seq_len, num_features))
        self.YY = np.empty((0, target_seq_len))
        self.sequence_lens = []
        self.seq_mask = np.empty((0, input_seq_len))

    def update(self, sku_id, inputs, targets, input_seq_len):
        self.sku_ids.append(sku_id)
        inputs_len = len(inputs)
        self.sequence_lens.append(inputs_len)

        # build current mask with zeros and ones
        cur_mask = np.zeros(input_seq_len)
        cur_mask[:inputs_len] = 1  # only the valid firsts should have the value of one

        xx_padded = np.pad(inputs, ((0, input_seq_len - inputs_len), (0, 0)), mode='constant', constant_values=0.)
        # here targets do NOT need to be padded because we do not have a sequence to sequence model
        # yy_padded = np.pad(targets, (0, series_max_len - len(targets)), mode='constant', constant_values=0.)

        assert len(xx_padded) == input_seq_len

        self.XX = np.vstack((self.XX, xx_padded[np.newaxis]))
        self.YY = np.vstack((self.YY, targets[np.newaxis]))

        self.seq_mask = np.vstack((self.seq_mask, cur_mask[np.newaxis]))

    def get_data(self, fraction=None, random_state=None):
        # from sklearn.model_selection import train_test_split
        skuIds, xx, yy, seqLens, seqMask = np.array(self.sku_ids), self.XX, self.YY, np.array(
            self.sequence_lens), self.seq_mask
        if fraction is None:
            return skuIds, xx, yy, seqLens, seqMask
        else:
            random_state = np.random if random_state is None else random_state

            cur_len = len(skuIds)
            assert cur_len == len(xx) and cur_len == len(yy) and cur_len == len(seqLens) and cur_len == len(seqMask)
            random_inds = random_state.choice(cur_len, int(cur_len * fraction))

            return skuIds[random_inds], xx[random_inds], yy[random_inds], seqLens[random_inds], seqMask[random_inds]

    def save(self, filepath, fraction=None, random_state=None):
        if fraction is None:
            np.savez(filepath, sku_ids=self.sku_ids, inputs=self.XX, targets=self.YY,
                     sequence_lengths=self.sequence_lens,
                     sequence_masks=self.seq_mask)
        else:
            skuIds, xx, yy, seqLens, seqMask = self.get_data(fraction=fraction, random_state=random_state)
            np.savez(filepath, sku_ids=skuIds, inputs=xx, targets=yy, sequence_lengths=seqLens, sequence_masks=seqMask)
