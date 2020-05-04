'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
class PValueSelector:
    def __init__(self, significance_level= 0.05):
        '''
            Setting up the algorithm
        :param significance_level: float, default = 0.05
            Used to compare the p-values with to find the features that must be removed from model
        '''
        self.significance_level = significance_level
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
        self.history = {}
        X_columns = [col for col in dataframe.columns if col != y_column]
        for col in X_columns:
            self.history[col] = []
        feature_state = list(np.ones(len(X_columns)))
        while True:
            self.iter +=1
            X_cols = self.bin_to_cols(feature_state, X_columns)
            X = dataframe[X_cols].values
            y = dataframe[y_column].values
            regressor_OLS = sm.OLS(y, X).fit()
            p_values = list(regressor_OLS.pvalues)
            p_values_index = 0
            for col in X_columns:
                if col in X_cols:
                    self.history[col].append(float(p_values[p_values_index]))
                    p_values_index+=1
                else:
                    self.history[col].append(-1)
            max_PValue = max(p_values)
            if max_PValue > self.significance_level:
                for j in range(len(X_cols)):
                    if regressor_OLS.pvalues[j].astype(float) == max_PValue:
                        feature_state[X_columns.index(X_cols[j])] = 0
            else:
                break
        self.selected_cols = self.bin_to_cols(feature_state, X_columns)
        return self.selected_cols

    def plot_process(self, title="P-Value Plot", save=False, file_path=None):
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