'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, LassoCV
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
class LassoSelector:
    def __init__(self, alpha_start : float = 0, alpha_finish : float = 2, n_alphas : int = 300, extend_step : float = 20, power : int = 2) -> None:
        '''
            Setting the algorithm
        :param alpha_start: float
            The starting point in the greedy search of coefficients
        :param alpha_finish: float
            The finish point in the greedy search of coefficients
        :param n_alphas: integer
            The number of points in greedy search
        :param extend_step: integer
            The quantity with which the :param alpha_start and :param alpha_finish will be updated
        :param power: integer
            Used to set a threshold in finding the best coefficients
        '''
        self.n_alphas = n_alphas
        self.alpha_start = alpha_start
        self.alpha_finish = alpha_finish
        self.extend_step = extend_step
        self.power = power

    def select(self, dataframe : 'pd.DataFrame', y_column : str, cv : int = 5) -> list:
        '''
            Selecting the most important columns
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we what to predict
        :param cv: integer
            Determines the cross-validation splitting strategy
        :return: list
            The list of columns selected by algorithm
        '''
        self.dataframe = dataframe
        self.y_column = y_column
        self.X_columns = [col for col in self.dataframe.columns if col != self.y_column]

        # Converting the data frame into numpy arrays
        X = self.dataframe[self.X_columns].values
        y = self.dataframe[self.y_column].values

        # Searching for the optimal value of alpha
        while True:
            # Generating the log search space for alpha
            self.alphas = np.logspace(self.alpha_start, self.alpha_finish, self.n_alphas, endpoint=True)

            # Creating and fitting up the lasso CV
            self.lasso_cv = LassoCV(cv=cv, alphas=self.alphas, random_state=0, tol=0.01)
            self.lasso_cv.fit(X, y)

            # Deciding if we need to expand the search space of not
            if self.lasso_cv.alpha_ == self.alphas[0]:
                self.alpha_start -= self.extend_step
            elif self.lasso_cv.alpha_ == self.alphas[-1]:
                self.alpha_finish += self.extend_step
            else:
                # Selecting the columns with the coefficients bigger then 10e-power
                self.choosed_cols = [self.X_columns[i] for i in range(len(self.lasso_cv.coef_)) if abs(self.lasso_cv.coef_[i])>10 **(-self.power)]
                return self.choosed_cols
    def plot_process(self, eps : float = 5e-3, title : str = "Lasso coef Plot", save : bool =False, file_path : str = None) -> None:
        '''
            Ploting the process of finding the best features
        :param eps: float
            Length of the path
        :param title string
            The title of the plot
        :param save boolean, default = False
            If the this parameter is set to False that the model will not save the model
            If it is set to True the plot will be saved using :param file_path
        :param file_path: string, default = None
            The file path where the plot will be saved
            If the :param save is set to False the it is not used
        :return:
            Plots the process of the algorithm
        '''
        # Converting the data frame into numpy arrays
        X = self.dataframe[self.X_columns].values
        y = self.dataframe[self.y_column].values

        # Generating the line space for the X axis
        alphas = np.linspace(self.lasso_cv.alpha_-0.1, self.lasso_cv.alpha_+0.1, self.n_alphas, endpoint=True)

        # Generating the alphas and coefficients
        alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False, alphas=alphas)
        neg_log_alphas_lasso = alphas_lasso
        max_coef = coefs_lasso[0][0]
        min_coef = coefs_lasso[0][0]

        # Plotting the lasso coefficients
        for i in range(len(coefs_lasso)):
            line_style = lambda col : '-' if col in self.choosed_cols else '--'
            plt.plot(neg_log_alphas_lasso, coefs_lasso[i], line_style(self.X_columns[i]), label=self.X_columns[i])
            if max(coefs_lasso[i]) > max_coef:
                max_coef = max(coefs_lasso[i])
            if min(coefs_lasso[i]) < min_coef:
                min_coef = min(coefs_lasso[i])
        plt.vlines(self.lasso_cv.alpha_, min_coef, max_coef, linestyles='dashed')
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title(title)
        plt.axis('tight')
        plt.legend()
        if save:
            plt.savefig(file_path)
        plt.show()