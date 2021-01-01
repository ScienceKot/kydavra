'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .errors import NoSuchMethodError

class LDAReducer:
    def __init__(self, solver : str = 'svd', method : str = 'pearson', min_corr : float = 0.5, max_corr : float = 0.8):
        '''
            The constructor of the reducer.
        :param solver: str, default = 'svd'
            The solver used by LDA Algorithm.
        :param method: str, default = 'person'
            The type of correlation used to find the most correlated features.
        :param min_corr: float, default = 0.5
            The minimal correlation between features accepted.
        :param max_corr: float, default = 0.8
            The maximal correlation between features accepted.
        '''
        # Setting up the Reducer
        if solver in ['svd', 'lsqr', 'eigen']:
            self.solver = solver
        else:
            raise NoSuchMethodError(f"kydavra doesn't sustain such solver as {solver}/nTry 'svd', 'lsqr' or 'eigen'")
        self.min_corr = min_corr
        self.max_corr = max_corr
        if method in ['pearson', 'kendall', 'spearman']:
            self.method = method
        else:
            raise NoSuchMethodError(f"kydavra doesn't sustain such method as {method}/nTry 'pearson', 'kendall' or 'spearman'")

    def reduce(self, dataframe : pd.DataFrame, target : str):
        '''
            This function reduces the correlated columns using LDA.
        :param dataframe: pandas DataFrame
            The pandas DataFrame on which the function is applied.
        :param target: str
            The target column name.
        :return: pandas DataFrame
            The data frame with reduced correlated columns.
        '''
        self.df = dataframe.copy()
        # Extracting X column names
        self.X_columns = [col for col in self.df.columns if col != target]

        # Calculating the correlation table
        self.corr_table = self.df[self.X_columns].corr(method=self.method).values

        # Searching for all correlated pairs.
        self.correlated_pairs = []
        for i in range(len(self.corr_table)):
            for j in range(i):
                if abs(self.corr_table[i][j]) < self.max_corr and abs(self.corr_table[i][j]) > self.min_corr:
                    self.correlated_pairs.append(tuple([self.X_columns[i], self.X_columns[j]]))

        # BUilding a set with unique correlated columns
        self.correlated_cols = set()
        for pair in self.correlated_pairs:
            self.correlated_cols.add(pair[0])
            self.correlated_cols.add(pair[1])

        # Grouping correlated columns into groups
        self.correlated_groups = dict()
        for col in list(self.correlated_cols):
            self.correlated_groups[col] = []
            for pair in self.correlated_pairs:
                if col in pair:
                    self.correlated_groups[col].append(pair[0] if pair[0] != col else pair[1])
        self.correlated_groups = [tuple([key] + self.correlated_groups[key]) for key in self.correlated_groups]
        self.correlated_groups = [tuple(sorted(group)) for group in self.correlated_groups]
        self.correlated_groups = list(set(self.correlated_groups))

        # Sorting groups to start
        self.correlated_groups.sort(key=len, reverse=True)
        self.ldas = dict()
        for group in self.correlated_groups:
            # Checking if columns are present in data frame.
            present = True
            for col in group:
                if col in self.df.columns:
                    pass
                else:
                    present = False
            if not present:
                break
            # Building the model in column group
            lda = LinearDiscriminantAnalysis(solver=self.solver)
            X = self.df[list(group)].values
            y = self.df[target]
            lda.fit(X, y)
            new_column_name = '_'.join(group)
            self.df[new_column_name] = lda.predict(X)
            self.df = self.df.drop(list(group), axis=1)
            self.ldas[group] = lda
        return self.df

    def apply(self, dataframe : pd.DataFrame):
        '''
            This functions applies the trained LDAs on the new data set.
        :param dataframe: pandas DataFrame
            The pandas DataFrame on which the function is applied.
        :return: pandas DataFrame
            The data frame with reduced correlated columns.
        '''
        # Applying the LDAs on the new data frame.
        self.df = dataframe.copy()
        for group in self.ldas:
            X = self.df[list(group)].values
            self.df['_'.join(group)] = self.ldas[group].predict(X)
            self.df = self.df.drop(list(group), axis=1)
        return self.df