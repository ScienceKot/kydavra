'''
Created with love by Sigmoid

@Author - Vladimir Stojoc - vladimir.stojoc@gmail.com
@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, enet_path
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class ElasticNetSelector():
    def __init__(self, alpha_start : float = -2, alpha_finish : float = 0, beta_start : float = -2, beta_finish : float = 0, n_alphas : int = 100, extend_step : int = 20,power : int = 2):
        '''
            Setting the algorithm
        :param alpha_start: float, by default = -2
            The starting point in the greedy search of coefficients for L1 regularization
        :param alpha_finish: float, by default = 1
            The finish point in the greedy search of coefficients for L1 regularization
        :param beta_start: float, by default = -2
            The starting point in the greedy search of coefficients for L2 regularization
        :param beta_finish: float, by default = 1
            The finish point in the greedy search of coefficients for L2 regularization
        :param n_alphas: integer, by default = 100
            The number of points in greedy search
        :param extend_step: integer, by default = 20
            The quantity with which the :param alpha_start and :param alpha_finish will be updated
        :param power: integer, by default = 2
            Used to set a threshold in finding the best coefficients
        '''
        self.n_alphas = n_alphas
        self.alpha_start = alpha_start
        self.alpha_finish = alpha_finish
        self.beta_start = beta_start
        self.beta_finish = beta_finish
        self.extend_step = extend_step
        self.power = power

    def select(self,dataframe : pd.DataFrame, y_column : str, cv : int = 5):
        '''
            Selecting the most important columns
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we what to predict
        :param cv: integer, by default = 5
            Determines the cross-validation splitting strategy
        :return: list
            The list of columns selected by algorithm
        '''
        #Splitting data
        self.dataframe = dataframe
        self.y_column = y_column
        self.X_columns = [col for col in self.dataframe.columns if col != self.y_column]
        X = self.dataframe[self.X_columns].values
        y = self.dataframe[self.y_column].values
        while True:
            #Setting alpha and beta values for greedy search
            self.alphas = np.logspace(self.alpha_start, self.alpha_finish, self.n_alphas, endpoint=True)
            self.betas = np.logspace(self.beta_start, self.beta_finish, self.n_alphas, endpoint=True)
            #Calculatina L1_ratio and alphas + reverse betas sum
            self.betas = self.betas[::-1]
            self.L1_ratio = self.alphas/(self.alphas+self.betas)
            self.alphas = self.alphas + self.betas 
            #Running ElasticNetCV
            self.enet_cv = ElasticNetCV(cv=cv, alphas=self.alphas, l1_ratio=self.L1_ratio, random_state=0, tol=0.0001)
            self.enet_cv.fit(X, y)
            #Enlarge the interval if the optimal value is on one of the endpoints and repeat cycle
            if self.enet_cv.alpha_ == self.alphas[0]:
                self.alpha_start -= self.extend_step
                self.beta_start -= self.extend_step
            elif self.enet_cv.alpha_ == self.alphas[-1]:
                self.alpha_finish += self.extend_step
                self.beta_finish += self.extend_step
            else:
                #Selecting cols witch are more useful
                self.choosed_cols = [self.X_columns[i] for i in range(len(self.enet_cv.coef_)) if abs(self.enet_cv.coef_[i])>10 **(-self.power)]
                return self.choosed_cols
    
    
    def plot_process(self, eps : float = 5e-3, title :str = "ElasticNet coef Plot", save : bool = False, file_path : str = None, regularization_plot : str = 'L1'):
        '''
            Ploting the process of finding the best features
        :param eps: float, by default = 5e-3
            Length of the path
        :param title string, by default = "ElasticNet coef Plot"
            The title of the plot
        :param save boolean, by default = False
            If the this parameter is set to False that the model will not save the model
            If it is set to True the plot will be saved using :param file_path
        :param file_path: string, by default = None
            The file path where the plot will be saved
            If the :param save is set to False the it is not used
        :param file_path: string, by default = "L1"
            Two possible values: "L1" and "L2"
            For "L1" shows the plot for changes in L1 alphas
            For "L2" shows the plot for changes in L2 betas                    
        :return:
            Plots the process of the algorithm
        '''
        X = self.dataframe[self.X_columns].values
        y = self.dataframe[self.y_column].values
        #Plotting for L1 regularization
        if regularization_plot == 'L1':
            alpha = self.enet_cv.alpha_ * self.enet_cv.l1_ratio_
            #Getting the nearest values around the best value of alpha
            alphas = np.linspace(alpha-0.1, alpha+0.1, self.n_alphas, endpoint=True)
            alphas_enet, coefs_enet,_ = enet_path(X, y, eps, fit_intercept=False, alphas=alphas) 
            neg_log_alphas_enet = alphas_enet
            max_coef = coefs_enet[0][0]
            min_coef = coefs_enet[0][0]
            #Plotting the importance for every feature of the data 
            for i in range(len(coefs_enet)):
                line_style = lambda col : '-' if col in self.choosed_cols else '--'
                plt.plot(neg_log_alphas_enet, coefs_enet[i], line_style(self.X_columns[i]), label=self.X_columns[i])
                if max(coefs_enet[i]) > max_coef:
                    max_coef = max(coefs_enet[i])
                if min(coefs_enet[i]) < min_coef:
                    min_coef = min(coefs_enet[i])
            plt.vlines(alpha, min_coef, max_coef, linestyles='dashed')
            plt.xlabel('-Log(alpha)')
        #Plotting for L1 regularization
        elif regularization_plot == "L2":
            beta =  self.enet_cv.alpha_ -  self.enet_cv.alpha_ * self.enet_cv.l1_ratio_
            #Getting the nearest values around the best value of beta
            betas = np.linspace(beta-0.1, beta+0.1, self.n_alphas, endpoint=True)
            betas_enet, coefs_enet, _ = enet_path(X, y, eps, fit_intercept=False, alphas=betas)
            neg_log_betas_enet = betas_enet
            max_coef = coefs_enet[0][0]
            min_coef = coefs_enet[0][0]
            #Plotting the importance for every feature of the data
            for i in range(len(coefs_enet)):
                line_style = lambda col : '-' if col in self.choosed_cols else '--'
                plt.plot(neg_log_betas_enet, coefs_enet[i], line_style(self.X_columns[i]), label=self.X_columns[i])
                if max(coefs_enet[i]) > max_coef:
                    max_coef = max(coefs_enet[i])
                if min(coefs_enet[i]) < min_coef:
                    min_coef = min(coefs_enet[i])
            plt.vlines(beta, min_coef, max_coef, linestyles='dashed')
            plt.xlabel('-Log(beta)')
        plt.ylabel('coefficients')
        plt.title(title)
        plt.axis('tight')
        plt.legend()
        if save:
            plt.savefig(file_path)
        plt.show()