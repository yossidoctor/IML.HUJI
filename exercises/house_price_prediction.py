import re
from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.express as px
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
    irrelevant_columns = ['id', 'date', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    positive_columns = ['price', 'sqft_living', 'sqft_lot', 'yr_built', 'bathrooms', 'floors']
    non_negative_columns = ['bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement', 'yr_renovated']
    x_df = X.drop(columns=irrelevant_columns).dropna()
    x_df = x_df[(x_df[positive_columns] > 0).all(axis=1)]
    x_df = x_df[(x_df[non_negative_columns] >= 0).all(axis=1)]
    x_df = x_df.loc[x_df['waterfront'].isin(range(2)) &
                    x_df['view'].isin(range(5)) &
                    x_df['condition'].isin(range(6)) &
                    x_df['grade'].isin(range(1, 15)) &
                    x_df['bedrooms'].isin(range(12))]
    x_df["recently_renovated"] = np.where(x_df["yr_renovated"] >= 2013, 1, 0)
    x_df = x_df.drop(columns=['price', 'yr_renovated'])
    return x_df if y is None else x_df, y.loc[y.index.isin(x_df.index)]


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
    for feature_name in X.columns:
        feature = X[feature_name]
        correlation = np.round(feature.cov(y) / (feature.std() * y.std()), 3)
        title = f"{correlation}-Pearson Correlation between {feature_name} Values and Price"
        px.scatter(pd.DataFrame({'x': feature, 'y': y}), x=feature, y=y, trendline="ols",
                   trendline_color_override='black', title=title,
                   labels={"x": f"{feature_name} values", "y": "Price"}) \
            .write_image(output_path + f"/{feature_name}_correlation.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df, df[['price']])

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_X, processed_train_y = preprocess_data(train_X, train_y)
    processed_test_X, processed_test_y = preprocess_data(test_X, test_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(processed_train_X, processed_train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
