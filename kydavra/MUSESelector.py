'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import pandas as pd
import math

class NotBinaryData(BaseException):
    pass

class MUSESelector():
    def __init__(self, num_features : int = 5, n_bins : int = 20, p : float = 0.2, T : float = 0.1) -> None:
        '''
            Setting up the algorithm.
        :param num_features: int, default = 5
            The number of features that will remain after the algorithm will be applied.
        :param n_bins: int, default = 20
            The number of bins in which the series will be split if it's continuous
        :param p: float, default = 0.2
            The minimal value for cumulated sum of probabilities of positive class frequency
        :param T: float, default = 0.1
            The minimal threshold of the class impurity for a bin to be passed to the selector
        '''
        self.__num_features = num_features
        self.__n_bins = n_bins
        self.__p = p
        self.__T = T

    def select(self, df : pd.DataFrame, target : str) -> list:
        '''
            Selecting the most important columns.
        :param df: pd.DataFrame
            The pandas DataFrame on which we want to apply feature selection
        :param target: str
            The column name of the value that we what to predict. Should be binary.
        :return: list
            The list of selected columns.
        '''
        # Checking if the target column is binary one
        self.df = df.copy()
        if len(self.df[target].unique()) != 2:
            raise NotBinaryData(f"{target} column isn't a binary column")

        # Creating an empty dictionary to store metadata about features
        features_data = dict()
        self.X_columns = [col for col in self.df.columns if col != target]
        self.selected_features = []

        # Defining the pandas numeric types of pandas
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in list(self.df.columns):
            # Ignoring target column
            if col == target:
                continue

            # Checking if column is numeric
            if self.df[col].dtype in numerics:
                # If column is numeric type we will apply bins split, else bins will be the values itself
                self.df[col] = pd.qcut(self.df[col], self.__n_bins, duplicates='drop', labels=False)

            # Defining the metadata about features
            labels = dict()
            labels['impurity'] = dict()
            labels['H'] = dict()
            labels['discarded'] = [].copy()

            # Calculating the metadata for every bin of the feature
            for q in self.df[col].unique():
                try:
                    labels['impurity'][q] = min(prob / len(self.df) for prob in dict(self.df[self.df[col]==q][target].value_counts()).values())
                except ValueError:
                    labels['impurity'][q] = 0
                try:
                    labels['H'][q] = sum(-p / len(self.df) * math.log2(p / len(self.df)) for p in dict(self.df[self.df[col]==q][target].value_counts()).values())
                except ValueError:
                    labels['H'][q] = 0
            features_data[col] = labels

        # Selecting the 'num_features' features
        for i in range(self.__num_features):
            # Defining the selecting criteria J
            J = dict()

            for feature in features_data.keys():
                # Skipping already selected features
                if feature in self.selected_features:
                    continue

                # Defining a empty list where selected bins will be stored
                selected_intervals = [].copy()
                H = features_data[feature]['H']

                # Selecting bins with the cumulated sum of p smaller than self.p
                p_sum = 0
                for key, value in {k: v for k, v in sorted(H.items(), key=lambda item: item[1])}.items():
                    if key in features_data[feature]['discarded']:
                        continue
                    selected_intervals.append(key)
                    p_sum += value
                    if p_sum > self.__p:
                        break

                # Calculating the J for feature
                J[feature] = sum(len(self.df[self.df[feature] == q])/len(self.df)*H[q] for q in selected_intervals)
            # Searching for the minimal value of J
            min_j_feature = list(J.keys())[0]
            for feature in J:
                if J[feature] < J[min_j_feature]:
                    min_j_feature = feature

            # Adding the feature with the minimal value of J to the selected_features
            self.selected_features.append(min_j_feature)

            # Discarding bins with the impurity smaller than self.T
            for feature in features_data:
                for j in features_data[feature]['impurity']:
                    if features_data[feature]['impurity'][j] < self.__T:
                        features_data[feature]['discarded'].append(j)
        return self.selected_features