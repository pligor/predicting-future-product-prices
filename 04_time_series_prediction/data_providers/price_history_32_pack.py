import numpy as np


class PriceHistoryPack(object):
    def __init__(self):
        super(PriceHistoryPack, self).__init__()
        self.sku_ids = []
        self.sequence_lens = []
        self.XX = []
        self.date_input_table = []
        self.seq_mask = []

    def update(self, sku_id, inputs, max_seq_len, date_inputs):
        self.sku_ids.append(sku_id)
        inputs_len = len(inputs)
        self.sequence_lens.append(inputs_len)

        # build current mask with zeros and ones
        cur_mask = np.zeros(max_seq_len)
        cur_mask[:inputs_len] = 1  # only the valid firsts should have the value of one

        xx_padded = np.pad(inputs, ((0, max_seq_len - inputs_len),), mode='constant', constant_values=0.)
        date_ins_padded = np.pad(date_inputs, ((0, max_seq_len - inputs_len),), mode='constant',
                                 constant_values=-1)  # empty string (invalid date info)

        assert len(xx_padded) == max_seq_len and len(date_ins_padded) == max_seq_len

        # TO BEWARE: prevent of using vstack because it is inefficient

        self.XX.append(xx_padded)
        self.date_input_table.append(date_ins_padded)

        self.seq_mask.append(cur_mask)

    def get_data(self, fraction=None, random_state=None):
        # from sklearn.model_selection import train_test_split
        skuIds, xx, seqLens, seqMask, date_ins = np.array(self.sku_ids), np.array(self.XX), np.array(
            self.sequence_lens), np.array(self.seq_mask), np.array(self.date_input_table)
        if fraction is None:
            return skuIds, xx, seqLens, seqMask, date_ins
        else:
            random_state = np.random if random_state is None else random_state

            cur_len = len(skuIds)
            assert cur_len == len(xx) and cur_len == len(seqLens) and cur_len == len(seqMask) \
                   and cur_len == len(date_ins)
            random_inds = random_state.choice(cur_len, int(cur_len * fraction))

            return skuIds[random_inds], xx[random_inds], seqLens[random_inds], seqMask[random_inds], date_ins[
                random_inds]

    def save(self, filepath, fraction=None, random_state=None):
        skuIds, xx, seqLens, seqMask, date_ins = self.get_data(fraction=fraction,
                                                               random_state=random_state)
        np.savez(filepath, sku_ids=skuIds, inputs=xx, sequence_lengths=seqLens, sequence_masks=seqMask,
                 date_inputs=date_ins)
