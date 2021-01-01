'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
# Importing all needed libraries
import numpy as np
import math

class ShannonSelector:
    def __init__(self, select_strategy : str = 'mean', nan_acceptance_level : float = 0.5) -> None:
        '''
            Setting up the algorithm
        :param select_strategy: str, default = 'mean'
            It defines the metric used to select the columns:
                'mean' : Use the mean value of information gain to select features
                'median' : Use the median value of information gain to select features
        :param nan_acceptance_level: float, default = 0.5
            The minimal quantity of NaN value in the column for feature to be accepted
        '''
        if select_strategy == 'mean':
            self.__select_strategy = np.mean
        elif select_strategy == 'median':
            self.__select_strategy = np.median
        else:
            raise ValueError("ShannonSelector doesn't accepts such a strategy. Choose 'mean' of 'median'")
        self.__nan_acceptance_level = nan_acceptance_level

    def __entropy(self, *classes_prob : list) -> float:
        '''
            This function calculates the entropy
        :param classes_prob: tuple
            The table with the normalized frequency of classes
        :return: float
            The entropy of sample
        '''
        entropy = 0
        for cls_prob in classes_prob:
            entropy -= cls_prob * math.log2(cls_prob)
        return entropy

    def select(self, dataframe : 'pd.DataFrame', y_column : str) -> list:
        '''
            This function implements the Shannon selection on the data frame
        :param dataframe: pandas.DataFrame
            The pandas DataFrame on which the algorithm must be applied
        :param y_column: str
            The string name of the target column
        :return: list
            The list of column names that where selected
        '''
        # Creating an empty dictionary to store the information gain for every column
        self.information_gains_ = dict()

        # The list of all column names except the target column
        X_columns = [col for col in dataframe.columns if col != y_column]

        # The number of sample for every class in the data set
        target_dict = dict(dataframe[y_column].value_counts())

        # The tuple with the names of every class name in the data set
        unique_classes_names = tuple(dataframe[y_column].unique())

        # The tuple with te normalised frequency of the target classed for the whole data set
        classes_prob = tuple(target_dict[unique_classes_names[i]]/len(dataframe) for i in range(len(unique_classes_names)))

        # The entropy of the whole data set based on the target clasees
        self.general_entropy_ = self.__entropy(*classes_prob)

        # Iterating throw dataframe columns
        for column in X_columns:
            # The dictionary with the classes count for the NaN values in the column
            nan_class_prob = dict(dataframe[dataframe[column].isna()][y_column].value_counts())

            # Skipping the columns with NaN values only in one class
            if len(nan_class_prob) < len(unique_classes_names):
                continue

            # Calculating the entropy for the samples with NaN values
            entropy_nan = self.__entropy(*tuple(nan_class_prob[class_name] / dataframe[column].isna().sum()
                                         for class_name in unique_classes_names))

            # Skipping the columns with NaN values in no class
            no_nan_class_prob = dict(dataframe[dataframe[column].notna()][y_column].value_counts())
            if len(no_nan_class_prob) < len(unique_classes_names):
                continue

            # Calculating the entropy for the samples without NaN values
            entropy_no_nan = self.__entropy(*tuple(no_nan_class_prob[class_name] / dataframe[column].notna().sum()
                                            for class_name in unique_classes_names))

            # Calculating the information gain for this column
            self.information_gains_[column] = self.general_entropy_ - entropy_nan * (
                        dataframe[column].isna().sum() / len(dataframe)) - entropy_no_nan * (dataframe[column].notna().sum() / len(dataframe))
        # Defining the columns where we will store all the selected columns
        saved_column = []

        for column in X_columns:
            # Selecting the columns with the number of NaN values smaller then __nan_acceptance_level or with the information gain smaller that mean or median of the information gains
            if dataframe[column].isna().sum() < self.__nan_acceptance_level * len(dataframe) or column in self.information_gains_ and self.information_gains_[column] < self.__select_strategy(list(self.information_gains_.values())):
                saved_column.append(column)
        return saved_column
