'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class PValueSelector:
    def __init__(self, significance_level : float = 0.05) -> None:
        '''
            Setting up the algorithm
        :param significance_level: float, default = 0.05
            Used to compare the p-values with to find the features that must be removed from model
        '''
        self.significance_level = significance_level
        self.iter = 0

    def bin_to_cols(self, feature_state : list, X_column : list) -> list:
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
        # Defining the empty dictionary with the p-value history
        self.history = {}

        # Defining the list with names of columns except the predicted one
        X_columns = [col for col in dataframe.columns if col != y_column]
        for col in X_columns:
            self.history[col] = []

        # Defining the feature states
        feature_state = list(np.ones(len(X_columns)))
        while True:
            self.iter +=1

            # Extracting the selected columns
            X_cols = self.bin_to_cols(feature_state, X_columns)
            X = dataframe[X_cols].values
            y = dataframe[y_column].values

            # Fitting the OLS model from statsmodels
            regressor_OLS = sm.OLS(y, X).fit()

            # Getting the p-values from OLS model
            p_values = list(regressor_OLS.pvalues)
            p_values_index = 0

            # Adding p-values to history dictionary
            for col in X_columns:
                if col in X_cols:
                    self.history[col].append(float(p_values[p_values_index]))
                    p_values_index+=1
                else:
                    self.history[col].append(-1)

            # Choosing the max value of p-value
            max_PValue = max(p_values)

            # Erasing the column with the p-value equal with the max value of the p-value, if the max value is
            # higher than significance level
            if max_PValue > self.significance_level:
                for j in range(len(X_cols)):
                    if regressor_OLS.pvalues[j].astype(float) == max_PValue:
                        feature_state[X_columns.index(X_cols[j])] = 0
            else:
                break

        # Returning the chose columns.
        self.selected_cols = self.bin_to_cols(feature_state, X_columns)
        return self.selected_cols

    def plot_process(self, title : str = "P-Value Plot", save : bool = False, file_path : str = None) -> None:
        '''
            Ploting the process of finding the best features
        :param title string
            The title of the plot
        :param save: boolean, default = False
            If the this parameter is set to False that the model will not save the model
            If it is set to True the plot will be saved using :param file_path
        :param file_path: string, default = None
            The file path where the plot will be saved
            If the :param save is set to False the it is not used
        '''
        for col in self.history:
            line_style = lambda col: '-' if col in self.selected_cols else '--'
            plt.plot(range(self.iter), self.history[col], line_style(col), label=col)
        plt.hlines(self.significance_level, xmin=0, xmax=self.iter, colors='k')
        plt.xlabel("Iterations")
        plt.ylabel("p-values")
        plt.legend(ncol=3)
        plt.title(title)
        if save:
            plt.savefig(file_path)
        plt.show()