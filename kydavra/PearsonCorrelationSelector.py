'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
from .errors import NotBetweenZeroAndOneError

class PearsonCorrelationSelector:
    def __init__(self, min_corr : float = 0.5, max_corr : float = 0.8, erase_corr : bool = False) -> None:
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
            if str(type(erase_corr)) != "<class 'bool'>":
                raise ValueError
        except NotBetweenZeroAndOneError:
            print("Min or Max Correlations are not seted between 0 and 1!")
            quit()
        except ValueError:
            print("Parameter erase_col isn't set as a boolean value!")
            quit()
        finally:
            self.min_corr = min_corr
            self.max_corr = max_corr
            self.erase_corr = erase_corr
    def index_to_cols(self, index_list : list) -> list:
        '''
            Converting the list of indexes in a list of features that will be picked by the model
        :param index_list: list
            A list of Indexes that should be converted into a list of feature names
        :return: list
            A list with feature names
        '''
        return [self.X_columns[i] for i in index_list]
    def select(self, dataframe : 'pd.Dataframe', y_column : str) -> list:
        '''
            Selecting the most important features
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The column name of the value that we what to predict
        :return: list
            The list of features that are selected by the algorithm as the best one
        '''
        # Getting the list with names of columns without the target one
        self.X_columns = [col for col in dataframe.columns if col != y_column]

        # Creating an empty list for correlated columns
        correlated_indexes = []

        # Getting the y-column index
        y_column_index = list(dataframe.columns).index(y_column)

        # Generating the correlation matrix
        self.corr_table = dataframe.corr()
        corr_matrix = self.corr_table.values

        # Searching for the columns correlated with the y-column
        for i in range(len(corr_matrix[y_column_index])):
            if abs(corr_matrix[y_column_index][i]) > self.min_corr and abs(corr_matrix[y_column_index][i]) < self.max_corr:
                correlated_indexes.append(i)

        # Creating a list with the names of columns correlated with the y-column
        self.correlated_cols = self.index_to_cols(correlated_indexes)

        # If we chose to erase a column from the correlated pairs we erase one of them
        if self.erase_corr:
            cols_to_remove = []
            for i in correlated_indexes:
                for j in correlated_indexes:
                    if abs(corr_matrix[i][j]) > self.max_corr and abs(corr_matrix[i][j]) < self.max_corr:
                        cols_to_remove.append(self.X_columns[i])
            cols_to_remove = set(cols_to_remove)

            # Removing the chose columns
            for col in cols_to_remove:
                self.correlated_cols.remove(col)
        return self.correlated_cols

