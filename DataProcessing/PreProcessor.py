import pandas as pd

from Config.Constants import Y_AXIS


class PreProcessor:
    """
    Inputs CSV data into pandas DF
    Removes Duration column and any non-binary attributes (i.e 'contact')
    Feature engineers remaining y/no mappings to binary 0/1
    For data schema please see: https://archive.ics.uci.edu/dataset/222/bank+marketing
    """

    @staticmethod
    def preprocess_csv(file_path:str):
        df = pd.read_csv(file_path, sep=";", quotechar='"')
        # drop duration - in documentation
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