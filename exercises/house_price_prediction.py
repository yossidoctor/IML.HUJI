import re
from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.io as pio

from IMLearn.utils import split_train_test  # todo: validate import

pio.templates.default = "simple_white"

# YYYYMMDDT000000, where 2100>YYYY>1900
date_pattern = re.compile(
    r'^(?:19[0-9][0-9]|20[0-9][0-9])(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])T000000')
positive_numerical_pattern = re.compile(r'^[1-9]\d*\.?\d*$|^0?\.\d*[1-9]\d*')
positive_int_pattern = re.compile(r'^[1-9]\d*')
year_built_pattern = re.compile(r'^(?:19[0-9][0-9]|20[0-9][0-9])')
year_renovated_pattern = re.compile(r'^(?:0|19[0-9][0-9]|20[0-9][0-9])')  # 0 or 2100>YYYY>1900


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    x_df = drop_irrelevant_features(X)
    x_df = validate_values(x_df)
    x_df = remove_old_listings(x_df)
    y_df = y.loc[y.index.isin(x_df.index)]  # Remove the samples from y
    X, y = x_df, y_df  # Update original X, y


def drop_irrelevant_features(x):
    return x.drop(columns=['zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'])


def remove_old_listings(x_df):
    """Keep only the most recent listing for each house (identified by id)"""
    x_df['date'] = pd.to_datetime(x_df['date'].str[:-7], format='%Y%m%d')
    x_df = x_df.sort_values(['id', 'date'], ascending=[True, False])
    x_df = x_df.drop_duplicates(subset=['id'], keep='first')
    return x_df


def validate_values(x_df):
    x_df = x_df.loc[
        x_df['date'].apply(lambda x: bool(date_pattern.match(x))) &
        x_df['price'].apply(lambda x: bool(positive_numerical_pattern.match(str(x)))) &
        x_df['bedrooms'].apply(lambda x: bool(positive_int_pattern.match(str(x)))) &
        x_df['bathrooms'].apply(lambda x: bool(positive_numerical_pattern.match(str(x)))) &
        x_df['sqft_living'].apply(lambda x: bool(positive_numerical_pattern.match(str(x)))) &
        x_df['sqft_lot'].apply(lambda x: bool(positive_numerical_pattern.match(str(x)))) &
        x_df['floors'].apply(lambda x: bool(positive_numerical_pattern.match(str(x)))) &
        x_df['yr_built'].apply(lambda x: bool(year_built_pattern.match(str(x)))) &
        x_df['yr_renovated'].apply(lambda x: bool(year_renovated_pattern.match(str(x))))]
    return x_df.dropna()


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df, df[['price']])
    preprocess_data(train_X, train_y)

    # Question 2 - Preprocessing of housing prices dataset
    # raise NotImplementedError()

    # Question 3 - Feature evaluation with respect to response
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
