import numpy as np


class PriceHistory27Pack(object):
    def __init__(self, input_seq_len, target_seq_len):
        super(PriceHistory27Pack, self).__init__()
        # self.XX = np.empty((0, input_seq_len))
        # self.date_input_table = np.empty((0, input_seq_len))
        # self.YY = np.empty((0, target_seq_len))
        # self.date_target_table = np.empty((0, target_seq_len))
        # self.seq_mask = np.empty((0, input_seq_len))
        self.sku_ids = []
        self.sequence_lens = []
        self.XX = []
        self.date_input_table = []
        self.YY = []
        self.date_target_table = []
        self.seq_mask = []

    def update(self, sku_id, inputs, targets, input_seq_len, date_inputs, date_targets):
        self.sku_ids.append(sku_id)
        inputs_len = len(inputs)
        self.sequence_lens.append(inputs_len)

        # build current mask with zeros and ones
        cur_mask = np.zeros(input_seq_len)
        cur_mask[:inputs_len] = 1  # only the valid firsts should have the value of one

        # xx_padded = np.pad(inputs, ((0, input_seq_len - inputs_len), (0, 0)), mode='constant', constant_values=0.)
        # date_ins_padded = np.pad(date_inputs, ((0, input_seq_len - inputs_len), (0, 0)), mode='constant',
        #                          constant_values='')  # empty string (invalid input)
        xx_padded = np.pad(inputs, ((0, input_seq_len - inputs_len),), mode='constant', constant_values=0.)
        date_ins_padded = np.pad(date_inputs, ((0, input_seq_len - inputs_len),), mode='constant',
                                 constant_values='')  # empty string (invalid input)

        assert len(xx_padded) == input_seq_len and len(date_ins_padded) == input_seq_len

        # TO BEWARE: prevent of using vstack because it is inefficient

        # self.XX = np.vstack((self.XX, xx_padded[np.newaxis]))
        # self.date_input_table = np.vstack((self.date_input_table, date_ins_padded[np.newaxis]))
        # self.XX.append(xx_padded[np.newaxis])
        # self.date_input_table.append(date_ins_padded[np.newaxis])
        self.XX.append(xx_padded)
        self.date_input_table.append(date_ins_padded)

        # we consider that the targets always have the same length (we are trying to predict the next 30 days)
        # that is why it is an easier case
        # self.YY = np.vstack((self.YY, targets[np.newaxis]))
        # self.date_target_table = np.vstack((self.date_target_table, date_targets[np.newaxis]))
        # self.YY.append(targets[np.newaxis])
        # self.date_target_table.append(date_targets[np.newaxis])
        self.YY.append(targets)
        self.date_target_table.append(date_targets)

        # self.seq_mask = np.vstack((self.seq_mask, cur_mask[np.newaxis]))
        # self.seq_mask.append(cur_mask[np.newaxis])
        self.seq_mask.append(cur_mask)

    def get_data(self, fraction=None, random_state=None):
        # from sklearn.model_selection import train_test_split
        skuIds, xx, yy, seqLens, seqMask, date_ins, date_tars = np.array(self.sku_ids), np.array(self.XX), np.array(
            self.YY), np.array(self.sequence_lens), np.array(self.seq_mask), np.array(self.date_input_table), np.array(
            self.date_target_table)
        if fraction is None:
            return skuIds, xx, yy, seqLens, seqMask, date_ins, date_tars
        else:
            random_state = np.random if random_state is None else random_state

            cur_len = len(skuIds)
            assert cur_len == len(xx) and cur_len == len(yy) and cur_len == len(seqLens) and cur_len == len(seqMask) \
                   and cur_len == len(date_ins) and cur_len == len(date_tars)
            random_inds = random_state.choice(cur_len, int(cur_len * fraction))

            return skuIds[random_inds], xx[random_inds], yy[random_inds], seqLens[random_inds], seqMask[random_inds], \
                   date_ins[random_inds], date_tars[random_inds]

    def save(self, filepath, fraction=None, random_state=None):
        skuIds, xx, yy, seqLens, seqMask, date_ins, date_tars = self.get_data(fraction=fraction,
                                                                              random_state=random_state)
        np.savez(filepath, sku_ids=skuIds, inputs=xx, targets=yy, sequence_lengths=seqLens, sequence_masks=seqMask,
                 date_inputs=date_ins, date_targets=date_tars)
