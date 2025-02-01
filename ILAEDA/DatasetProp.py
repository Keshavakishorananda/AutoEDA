import pandas as pd
import numpy as np

class DatasetProp():
    def __init__(self, Dataset):
        self.Dataset = Dataset

        self.df = pd.read_csv(self.Dataset, sep='\t', index_col=0)
        self.filter_options = {}
        self.non_unique_filter_options = {}
        self.filter_list = []

        # Get the columns
        column_mask = pd.array(self.df.nunique(axis=0) > 1)
        self.Keys = self.df.iloc[:, column_mask].columns.tolist()

        # Get the numeric columns
        self.numeric_keys = self.df[self.Keys].select_dtypes(np.number).columns.tolist()

        # Get filter_list
        for key in self.Keys:
            self.non_unique_filter_options[key] = self.df[key].dropna(
            ).tolist()
            self.filter_options[key] = self.df[key].dropna(
            ).unique().tolist()
            self.filter_list.extend(self.filter_options[key])

        bins = [0] + [0.1 / 2 ** i for i in range(11, 0, -1)] + [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                                 0.5, 0.55,
                                                                 0.6, 0.7, 0.8, 0.9, 1.0]
        self.bins = np.array(bins)
    