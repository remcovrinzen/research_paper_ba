# MIT License
#
# Copyright(c) 2018 Remco Vrinzen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sklearn.preprocessing as skpp
import pandas as pd


def create_labels_from_y_values(y_values):
    """
    Return y labels and labels object 

    Parameters
    ----------
    y_values: numpy array

    Returns
    -------
    labelled y_values, Label encoder

    """
    le = skpp.LabelEncoder()
    label_y = le.fit_transform(y_values)

    return label_y, le


def transform_values_into_categorical(df, categorical_vars):
    """
    Return df with categorized variables

    Parameters
    ----------
    df: Pandas dataframe
    categorical_vars: array with var names

    Returns
    -------
    Df with dummy columns for categorical variables

    """
    for var in categorical_vars:
        df_dummy = pd.get_dummies(df[var], prefix=var)
        df = df.drop([var], axis=1)
        df = pd.concat((df, df_dummy), axis=1)

    return df


def normalize_numeric_variables(df, numeric_vars):
    """
    Return df with normalized variables

    Parameters
    ----------
    df: Pandas dataframe
    numeric_vars: array with var names

    Returns
    -------
    Df with normalized columns for numerical variables

    """
    pass


def create_new_column_with_difference_between_2(df, column1, column2):
    """
    Return df with new delta column: column1 - column2

    Parameters
    ----------
    df: Pandas dataframe
    column1: Column string
    column2: Column string

    Returns
    -------
    Df with added delta column

    """

    df['delta_' + column1 + "_" + column2] = df[column1] - df[column2]

    return df
