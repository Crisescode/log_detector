# coding: utf-8

import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from collections import OrderedDict


class DataLoader(object):
    def __init__(self, train_csv: str, label_csv: str, split_ratio: float):
        self.train_file = train_csv
        self.label_file = label_csv
        self.split_ratio = split_ratio

    def run(self):
        struct_log = pd.read_csv(self.train_file, engine='c', na_filter=False, memory_map=True)
        data_dict =OrderedDict()

        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        label_data = pd.read_csv(self.label_file, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index("BlockId")
        label_dict = label_data["Label"].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

        (x_train, y_train), (x_test, y_test) = self._split_data(
            data_df["EventSequence"].values, data_df["Label"].values, self.split_ratio
        )

        print(y_train.sum(), y_test.sum())

        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos

        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

        return (x_train, y_train), (x_test, y_test)

    def _split_data(self, x_data, y_data=None, train_ratio: float = 0):
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]

        if y_data is None:
            raise
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]

        indexes = shuffle(np.arange(x_train.shape[0]))
        x_train = x_train[indexes]
        y_train = y_train[indexes]

        return (x_train, y_train), (x_test, y_test)



