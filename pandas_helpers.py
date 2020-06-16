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

import numpy as np
import os
import pandas as pd


def get_average_value_series(series):
    """
    Return mean value of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Mean of series

    """
    return np.mean(series)


def get_std_series(series):
    """
    Return standard deviation value of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Deviation of series

    """
    return np.std(series)


def get_range_series(series):
    """
    Return range of values of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Range of series

    """
    return np.max(series) - np.min(series)


def get_min_series(series):
    """
    Return minimum of values of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Minimum of series

    """
    return np.min(series)


def get_max_series(series):
    """
    Return maximum of values of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Maximum of series

    """
    return np.max(series)


def get_number_of_unique_values_series(series):
    """
    Return number of unique values of Pandas series

    Parameters
    ----------
    series: (Numeric) Pandas series

    Returns
    -------
    Number of unique values of series

    """
    return series.unique().shape[0]


def set_df_name_if_missing(df):
    """
    Sets dataframe name if missing

    Parameters
    ----------
    df : Pandas dataframe

    """

    try:
        df.name
    except:
        df.name = "Dataframe"


def print_console_title(title):
    """
    Prints clear console title

    Parameters
    ----------
    title : String

    """

    print(title.center(80, '-'))


def get_dtypes(df):
    """
    Returns all dtypes of variables in dataframe

    Parameters
    ----------
    df: Pandas dataframe or series

    Returns
    -------

    Dataframe with all dtypes per column

    """

    return df.dtypes


def nan_summary(df, printing=False):
    """
    Return summary of columns with nan values

    Parameters
    ----------
    df : Pandas dataframe
    printing: Boolean

    Returns
    -------
    String of frame with all columns with NaN values and the percentage NaN

    """
    set_df_name_if_missing(df)

    percentage_missing = round(
        df.isnull().sum() / get_number_of_rows(df), 2) * 100

    only_missing_columns = percentage_missing[percentage_missing != 0]

    if printing:
        print((' NAN_SUMMARY ' + df.name).center(80, '-'))
        print(only_missing_columns.to_string())
        print("\n")

    return only_missing_columns.to_string()


def get_object_columns(df):
    """
    Return all columns with dtype object

    Parameters
    ----------
    df : Pandas dataframe

    Returns
    -------
    Dataframe with columns of dtype object

    """

    return df.select_dtypes(include=['object'])


def object_columns_summary(df):
    """
    Return summary about object columns in dataframe

    Parameters
    ----------
    df : Pandas dataframe

    Returns
    -------
    Dataframe with columns of dtype object

    """

    object_columns = get_object_columns(df)

    results_frame = pd.DataFrame(columns=['Variable', 'Number_of_unique'])

    for var_name in object_columns:
        results_frame = results_frame.append({
            'Variable': var_name,
            'Number_of_unique': get_number_of_unique_values_series(object_columns[var_name])
        }, ignore_index=True)

    return results_frame.to_string()


def get_unique_values_of_all_object_columns(df):
    """
    Returns dictionary with all unique values per column 

    Parameters
    ----------
    df : Pandas dataframe

    Returns
    -------
    Dictionary with unique values per column

    """

    object_columns = get_object_columns(df)

    results_dict = {}

    for var_name in object_columns:
        results_dict[var_name] = object_columns[var_name].unique()

    return results_dict


def numeric_value_summary(df, printing=False):
    """
    Return summary of numeric columns in dataframe

    Parameters
    ----------
    df : Pandas dataframe
    printing: Boolean

    Returns
    -------
    Pandas dataframe with info about the numeric value columns
    of the original Pandas dataframe

    """
    set_df_name_if_missing(df)

    numeric_columns = df.select_dtypes(include=[np.number])

    results_frame = pd.DataFrame(
        columns=['Variable', 'Min', 'Max', 'Range', 'Mean', 'Std'])

    for var_name in numeric_columns:
        results_frame = results_frame.append({'Variable': var_name,
                                              'Min': round(get_min_series(numeric_columns[var_name]), 2),
                                              'Max': round(get_max_series(numeric_columns[var_name]), 2),
                                              'Range': round(get_range_series(numeric_columns[var_name]), 2),
                                              'Mean': round(get_average_value_series(numeric_columns[var_name]), 2),
                                              'Std': round(get_std_series(numeric_columns[var_name]), 2)
                                              }, ignore_index=True)

    return results_frame.to_string()


def create_summary_file(df):
    """
    Creates a summary text file about the Pandas dataframe

    Parameters
    ----------
    df : Pandas dataframe

    """

    with open(os.getcwd() + "\summary.txt", 'w') as file:
        file.write("Rows:" + str(get_number_of_rows(df)) + "\n")
        file.write("Columns:" + str(get_number_of_columns(df)) + "\n")
        file.write("\n")

        file.write(get_dtypes(df).to_string() + "\n")
        file.write("\n")

        file.write("NaN summary \n")
        file.write(nan_summary(df) + "\n")
        file.write("\n")

        file.write("Numerical variables \n")
        file.write(numeric_value_summary(df) + "\n")
        file.write("\n")

        file.write("Object variables \n")
        file.write(object_columns_summary(df) + "\n")
        file.write("\n")

        unique_values_per_column = get_unique_values_of_all_object_columns(df)

        for column in unique_values_per_column:
            file.write(column + "\t" +
                       str(unique_values_per_column[column]) + "\n")
            file.write("\n")


def get_number_of_rows(data):
    """
    Returns the number of rows in the data

    Parameters
    ----------
    data: Pandas dataframe or series

    Returns
    -------
    number_of_rows

    """
    return data.shape[0]


def get_number_of_columns(data):
    """
    Returns the number of columns in the data

    Parameters
    ----------
    data: Pandas dataframe or series

    Returns
    -------
    number_of_columns

    """
    return data.shape[1]


def check_correlation_between_nan_values_2_columns(df, column1, column2):
    """

    Return a dataframe with the percentage values occuring in column 2 for each nan row of column1

    Parameters
    ----------
    df: Pandas dataframe
    column1: String
    column2: String

    Returns
    -------
    Pandas dataframe
    """

    missing_rows = df[df[column1].isnull()]

    nan_rows = get_number_of_rows(missing_rows)

    result = missing_rows[column2].value_counts() / nan_rows * 100

    return result


def get_fraction_value_of_total(series):
    """

    Return a dataframe with the fraction values of total

    Parameters
    ----------
    series: Pandas dataframe

    Returns
    -------
    Pandas series
    """

    value_counts = series.value_counts()

    return value_counts / value_counts.sum()
