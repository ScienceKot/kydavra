'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
from functools import reduce
from operator import and_, or_
import pandas as pd

class MixerSelector:
    def __init__(self, selectors : list, strategy : str = 'intersection') -> None:
        '''
            Setting up the algorithm.
        :param selectors: list
            The list of initialized selectors.
        :param strategy: str, default = 'intersection'
            If set to 'union' the selector will return union of selected columns returned by selectors
            If set to 'intersection' the selector will return the intersection of columns returned by selectors
        '''
        self.__selectors = selectors
        self.__strategy = strategy

    def select(self, df : pd.DataFrame, target : str) -> list:
        '''
            Selecting the most important columns.
        :param df: pd.DataFrame
            The pandas DataFrame on which we want to apply feature selection
        :param target: str
            The column name of the value that we what to predict. Should be binary.
        :return: list
            The list of selected columns.
        '''
        # Defining the list where sets of columns for every selector will be stored
        selected_columns = []
        # Filling up the list with the columns selected by every selector
        for selector in self.__selectors:
            selected_columns.append(set(selector.select(df, target)))

        # Applying the chose strategy
        if self.__strategy == 'union':
            # Returning the union of the selected columns
            return list(reduce(or_, selected_columns))
        elif self.__strategy == 'intersection':
            # Returning the intersection of the selected columns
            return list(reduce(and_, selected_columns))
        else:
            # Raising an error if the strategy isn't the union or intersection
            raise ValueError(f"No such strategy as {self.__strategy}!")
