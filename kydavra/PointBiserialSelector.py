'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import numpy as np
import pandas as pd
from math import sqrt
from .errors import NotBetweenZeroAndOneError
class PointBiserialCorrSelector:
    def __init__(self, min_corr : float = 0.5, max_corr : float = 0.8, last_level : int = 2) -> None:
        '''
            Setting up the algorithm
        :param min_corr: float, between 0 and 1, default = 0.5
            The minimal positive correlation value that must the feature have with y_column
        :param max_corr: float, between 0 and 1, default = 0.8
            The maximal positive correlation value that must the feature have with y_column
        :param erase_corr: boolean, default = False
            If set as False the selector doesn't erase features that are highly correlated between themselves
            If set as True the selector does erase features that are highly correlated between themselves
        '''
        try:
            if min_corr < 0 or max_corr > 1:
                raise NotBetweenZeroAndOneError
        except NotBetweenZeroAndOneError:
            print("Min or Max Correlations are not seted between 0 and 1!")
            quit()
        finally:
            self.min_corr = min_corr
            self.max_corr = max_corr
            self.last_level = last_level

    def pointBiserialCorr(self, dataframe : 'pd.DataFrame', X_col : str, y_col : str) -> float:
        '''
            Calculating the Point-Biserial Correlation
        :param dataframe: pandasDataFrame
            DataFrame from which we extract needed columns to calculate the correlation
        :param X_col: string
            The name of the column in DataFrame that holds continous data
        :param y_col: sting
            The name of the column in DataFrame that holds the binary Data
        :return:
            The calculated Point Biserial Correlation between X_col and y_col
        '''
        # Getting unique values from target column
        y_unique = dataframe[y_col].unique()

        # Sub-Setting data set
        subset0 = dataframe[dataframe[y_col] == y_unique[0]][X_col]
        subset1 = dataframe[dataframe[y_col] == y_unique[1]][X_col]

        # Getting the standard deviation for the column for the whole data set
        std = np.std(dataframe[X_col])

        # Getting the length of the sets
        n = len(dataframe[y_col])
        n0 = len(subset0)
        n1 = len(subset1)

        # Calculating the means of the both data set
        Mean0 = subset0.mean()
        Mean1 = subset1.mean()

        # Calculating the Point-Biserial correlation
        return (Mean0 - Mean1) * sqrt((n0*n1)/n**2) / std

    def select(self, dataframe : 'pd.DataFrame', y_column : str) -> list:
        '''
            Selecting the most important columns
        :param dataframe: pandas DataFrame
            Data Frame on which the algorithm is applied
        :param y_column: string
            The column name of the value that we what to predict
        :return: list
            The list of features that are selected by the algorithm as the best one
        '''
        # Defining the empty correlation and correlation-level dictionary
        corr_dict = {}
        corr_levels = {}

        # Creating the bountries list
        bountries = np.linspace(0, 1, 11, endpoint=True)

        # Getting the numerical columns
        numeric_cols_names = dataframe.select_dtypes(include=np.number).columns.tolist()

        # Calculating the point-biserial correlation for every numerical column
        for numeric_col in numeric_cols_names:
            if numeric_col != y_column:
                corr_dict[numeric_col] = self.pointBiserialCorr(dataframe, numeric_col, y_column)

        # Placing the columns in correlation levels
        for i in range(len(bountries)-1):
            corr_levels[str(bountries[i])] = []
            for numeric_col in corr_dict:
                if abs(corr_dict[numeric_col]) > bountries[i] and abs(corr_dict[numeric_col]) < bountries[i+1]:
                    corr_levels[str(bountries[i])].append(numeric_col)
        corr_levels = {x:corr_levels[x] for x in corr_levels if corr_levels[x]!=[]}
        levels = np.array(list(corr_levels.keys())).astype('float')

        # Getting the last_level highest correlated levels
        highest_levels = np.argsort(levels)[::-1][:self.last_level]
        self.high_corralated = []
        for level in highest_levels:
            self.high_corralated.extend(corr_levels[list(corr_levels.keys())[level]])

        # Returning the selected columns
        return self.high_corralated