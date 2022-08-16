# coding: utf-8

import re
import pandas as pd
from collections import OrderedDict


class DataLoader(object):
    def __init__(self, log_file: str, label_file: str, split_ratio: float, split_type: str):
        self.log_file = log_file
        self.label_file = label_file
        self.split_ratio = split_ratio
        self.split_type = split_type

    def run(self):
        struct_log = pd.read_csv(self.log_file, engine='c', na_filter=False, memory_map=True)
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
            data_df["EventSequence"].values, data_df["Label"].values, self.split_ratio, self.split_type
        )

        print(y_train.sum(), y_test.sum())


    def _split_data(self, x_data, y_data=None, train_ratio=0, split_type="uniform"):
        pass



