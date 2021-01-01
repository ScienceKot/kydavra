'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
@Author - Stojoc Vladimir - stojoc.vladimir@gmail.com
'''
import pandas as pd
from sklearn.decomposition import PCA

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class PCAFilter():
    def __init__(self, n_components : int = None):
        '''
            Setting the algorithm
        :param n_components: integer, by default = None
            Number of components to keep
        '''
        self.n_components = n_components

    def filter(self, dataframe : pd.DataFrame, y_column : str):
        '''
            Creating filter to new data and reducing the dimensionality of the data
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we have to predict
        '''
        #Splitting dataframe
        self.dataframe = dataframe.copy()
        self.y_column = y_column
        self.X_columns = [col for col in self.dataframe.columns if col != self.y_column]
        self.X = self.dataframe[self.X_columns].values
        self.y = self.dataframe[y_column].values
        #Creating filter
        self.pca = PCA(n_components = self.n_components)
        self.pca.fit(self.X)
        #Creating new data based on the filter
        X_pca = self.pca.transform(self.X)
        X_new = self.pca.inverse_transform(X_pca)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)
        #Create and return new Dataframe
        X_new[y_column] = self.y
        return X_new

    def apply(self, dataframe : pd.DataFrame):
        '''
            Reducing the dimensionality of the data
            based on an already existed filter
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we have to predict
        '''
        #Splitting data
        self.dataframe = dataframe.copy()
        X_columns = [col for col in self.dataframe.columns if col != self.y_column]
        X = self.dataframe[X_columns].values
        y = self.dataframe[self.y_column].values
        #Applying filter to the new dataframe
        X_pca = self.pca.transform(X)
        X_new = self.pca.inverse_transform(X_pca)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)
        #Create and return new Dataframe
        X_new[self.y_column] = y
        return X_new