'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
import pandas as pd
import math

def memoize(fun: 'function') -> 'function':
    '''
        The cache function.
    :param fun: function
        Function on which the decorator is applyed.
    :return: function
    '''
    results = dict()

    # Defining the internal helper function
    def helper(p: float):
        if p not in results:
            results[str(p)] = fun(p)
        return results[str(p)]

    return helper


@memoize
def info_theory(p: float) -> float:
    '''
        The information theory function.
    :param p: float
        The probability passed to function,
    :return: float
        The entropy.
    '''
    if p == 0:
        return 0
    else:
        return -p * math.log2(p)


class M3USelector:
    def __init__(self, num_features : int = 5, n_bins : int = 20) -> None:
        '''
            The Selector's constructor.
        :param num_features: int, default = 5
            The number of feature selected.
        :param n_bins: int, default = 20
            The number of bins to apply on the continous feature.
        '''
        self.__num_features = num_features
        self.__n_bins = n_bins

    def select(self, dataframe : pd.DataFrame, target : str) -> list:
        '''
            The selector function
        :param dataframe: pandas DataFrame
            The pandas DataFrame on which the function is applied.
        :param target: str
            The target column name.
        :return:
        '''
        self.df = dataframe.copy()
        self.selected_features = []
        self.X_columns = [col for col in list(self.df.columns) if col != target]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in list(self.df.columns):
            # Ignoring target column
            if col == target:
                continue

            # Checking if column is numeric
            if self.df[col].dtype in numerics:
                # If column is numeric type we will apply bins split, else bins will be the values itself
                self.df[col] = pd.qcut(self.df[col], self.__n_bins, duplicates='drop', labels=False)
        for i in range(self.__num_features):
            # Defining the Minimal Mean Uncertainty Dictionary
            M2U = dict()

            # For every column wit already selected columns we calculates the Uncertainty
            for col in self.X_columns:
                if col in self.selected_features:
                    continue
                U = dict()
                MU = dict()
                temp = self.selected_features + [col]

                # Calculating the uncertainty for every column
                for temp_col in temp:
                    U[temp_col] = dict()
                    for q in self.df[temp_col].unique():
                        U[temp_col][q] = sum(info_theory(p) for p in [len(self.df[(self.df[temp_col] == q) & (self.df[target].isin(cls))]) / len(self.df)
                                                                           for cls in [[self.df[target].mode()], list(self.df[self.df[target] != self.df[target].mode()[0]][target])]])
                for temp_col in temp:
                    MU[temp_col] = min(U[temp_col].values())
                M2U[col] = sum(MU.values()) / len(MU)
            selected_col = M2U.keys()

            # Selecting the column with the minimal mean minimal uncertainty
            for col in M2U:
                if M2U[col] == min(M2U.values()):
                    selected_col = col
            self.selected_features.append(selected_col)
        return self.selected_features