'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
class ChiSquaredSelector:
    def __init__(self, significance_level=0.05):
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
        X_columns = [col for col in dataframe.columns if col != y_column]
        self.chi2_history = {}
        self.p_value_history = {}
        for col in X_columns:
            self.chi2_history[col] = []
            self.p_value_history[col] = []
        feature_state = list(np.ones(len(X_columns)))
        while True:
            self.iter +=1
            X_cols = self.bin_to_cols(feature_state, X_columns)
            X = dataframe[X_cols].values
            y = dataframe[y_column].values
            chi2_vals, p_vals = chi2(X, y)
            index = 0
            for col in X_columns:
                if col in X_cols:
                    self.chi2_history[col].append(float(chi2_vals[index]))
                    self.p_value_history[col].append(float(p_vals[index]))
                    index+=1
                else:
                    self.chi2_history[col].append(-1)
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

    def plot_chi2(self, title="Chi2 Plot", save=False, file_path=None):
        '''
            Plotting the process
        :param title
            The title of the plot
        :param save: boolean, default = False
            If the this parameter is set to False that the model will not save the model
            If it is set to True the plot will be saved using :param file_path
        :param file_path: string, default = None
            The file path where the plot will be saved
            If the :param save is set to False the it is not used
        '''
        for col in self.chi2_history:
            line_style = lambda col: '-' if col in self.choosed_cols else '--'
            plt.plot(range(self.iter), self.history[col], line_style(col), label=col)
        plt.xlabel("Iterations")
        plt.ylabel("Chi2 Score")
        plt.legend(ncol=3)
        plt.title(title)
        if save:
            plt.savefig(file_path)
        plt.show()

    def plot_p_value(self, title="P-Value Plot", save=False, file_path=None):
        '''
        :param save: boolean, default = False
            If the this parameter is set to False that the model will not save the model
            If it is set to True the plot will be saved using :param file_path
        :param file_path: string, default = None
            The file path where the plot will be saved
            If the :param save is set to False the it is not used
        '''
        line_style = lambda col: '-' if col in self.choosed_cols else '--'
        for col in self.chi2_history:
            plt.plot(range(self.iter), self.p_value_history[col], line_style(col), label=col)
        plt.xlabel("Iterations")
        plt.ylabel("p-values")
        plt.legend(ncol=3)
        plt.title(title)
        if save:
            plt.savefig(file_path)
        plt.show()