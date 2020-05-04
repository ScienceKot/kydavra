'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import numpy as np
from sklearn.feature_selection import f_classif, f_regression
class ANOVASelector:
    def __init__(self, significance_level=0.05, classification = True):
        '''
            Setting up the algorithm
        :param significance_level: float, default = 0.05
            Used to compare the p-values with to find the features that must be removed from model
        :param classification
            Set the algorithm on a classification problem or on a regression problem
        '''
        self.significance_level = significance_level
        self.classification = classification
        self.iter = 0
    def bin_to_cols(self, feature_state, X_column):
        '''
            Converting the binary state list into names of features that are included in model
        :param feature_state: list
            A list with zeros and ones used by model to understand what values should be pick by the model
            (when value = 1) and what it shouldn't pick (when value = 0)
        :param X_column: list
            The list with features of the dataframe before the algorithm is applied
        :return: list
            The list of features that where picked by the algorithm
        '''
        included_columns = [X_column[i] for i in range(len(feature_state)) if feature_state[i]==1]
        return included_columns

    def select(self, dataframe, y_column):
        '''
            Selecting the most important columns
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we what to predict
        :return: list
            The list of features that are selected by the algorithm as the best one
        '''
        X_columns = [col for col in dataframe.columns if col != y_column]
        self.F_history = {}
        self.p_value_history = {}
        for col in X_columns:
            self.F_history[col] = []
            self.p_value_history[col] = []
        feature_state = list(np.ones(len(X_columns)))
        while True:
            self.iter +=1
            X_cols = self.bin_to_cols(feature_state, X_columns)
            X = dataframe[X_cols].values
            y = dataframe[y_column].values
            if self.classification:
                F_vals, p_vals = f_classif(X, y)
            else:
                F_vals, p_vals = f_regression(X, y)
            index = 0
            for col in X_columns:
                if col in X_cols:
                    self.F_history[col].append(float(F_vals[index]))
                    self.p_value_history[col].append(float(p_vals[index]))
                    index+=1
                else:
                    self.F_history[col].append(-1)
                    self.p_value_history[col].append(-1)
            max_PValue = max(p_vals)
            if max_PValue > self.significance_level:
                for j in range(len(X_cols)):
                    if p_vals[j].astype(float) == max_PValue:
                        feature_state[X_columns.index(X_cols[j])] = 0
            else:
                break
        self.choosed_cols = self.bin_to_cols(feature_state, X_columns)
        return self.choosed_cols