from pathlib import Path

import pandas as pd
from pandas import DataFrame

from Config.Constants import Y_AXIS, INPUT_DATA_DIR


class PreProcessor:
    """
    Inputs CSV data into pandas DF
    Removes Duration column and any non-binary attributes (i.e 'contact')
    Feature engineers remaining y/no mappings to binary 0/1
    For data schema please see: https://archive.ics.uci.edu/dataset/222/bank+marketing
    """

    """
    Chunk Preprocess for larger data sets in blob storage 
    Use this for Production Pipelines
    """
    # Defensability analysis:
    # formatting issue, what if the no/yes mapping is incorrect
    # doesn't make sense to retry, but what do we do
    # drop the entire chunk? or is there a way to find the 'offending rows' and retry the remaining rows within the chunk
    @staticmethod
    def preprocess_chunk(raw_df: DataFrame) -> DataFrame:
        # drop duration - Specified in dataset documentation: https://archive.ics.uci.edu/dataset/222/bank+marketing
        raw_df = raw_df.drop('duration', axis = 1)

        # convert y/no to binary

        raw_df[Y_AXIS] = raw_df[Y_AXIS].map({'no': 0, 'yes': 1})
        raw_df['default'] = raw_df['default'].map({'no': 0, 'yes': 1})
        raw_df['housing'] = raw_df['housing'].map({'no': 0, 'yes': 1})
        raw_df['loan'] = raw_df['loan'].map({'no': 0, 'yes': 1})

        features = ['age', 'balance', 'campaign', 'pdays', 'previous', 'default', 'housing', 'loan']
        X = raw_df[features]
        y = raw_df[Y_AXIS]

        numeric_df = pd.concat([X, y], axis=1)
        numeric_df.drop('y', axis=1, inplace= True)

        return numeric_df

    """
    Static Preprocess entire file on Disk
    Use this for LocalPipeline
    """
    @staticmethod
    def preprocess_csv(file_path: Path) -> DataFrame:
        df = pd.read_csv(file_path, sep=";", quotechar='"')
        # drop duration - Specified in dataset documentation: https://archive.ics.uci.edu/dataset/222/bank+marketing
        df = df.drop('duration', axis = 1)

        # convert y/no to binary

        df[Y_AXIS] = df[Y_AXIS].map({'no': 0, 'yes': 1})
        df['default'] = df['default'].map({'no': 0, 'yes': 1})
        df['housing'] = df['housing'].map({'no': 0, 'yes': 1})
        df['loan'] = df['loan'].map({'no': 0, 'yes': 1})

        features = ['age', 'balance', 'campaign', 'pdays', 'previous', 'default', 'housing', 'loan']
        X = df[features]
        y = df[Y_AXIS]

        numeric_df = pd.concat([X, y], axis=1)

        return numeric_df