#!/usr/bin/env python3
'''
    Function def from_numpy(array):
    that creates a pd.DataFrame from a np.ndarray
'''

import pandas as pd
import string

def from_numpy(array):
    '''
        Function def from_numpy(array):
        that creates a pd.DataFrame from a np.ndarray

        Args:
            - array is the np.ndarray from which you should
            create the pd.DataFrame
            - The columns of the pd.DataFrame should be labeled
            in alphabetical order and capitalized.

        Returns:
            - Returns: the newly created pd.DataFrame
    '''
    if array.shape[1] > 26:
        raise ValueError("Array has more than 26 columns.")

    columns = list(string.ascii_uppercase[:array.shape[1]])

    # Create the DataFrame
    df = pd.DataFrame(array, columns=columns)

    return df
