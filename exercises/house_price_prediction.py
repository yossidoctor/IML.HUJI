from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"
# TODO: REPLACE ALL THE .WRITE_IMAGE TO SHOW, AFTER FINISHING THE EXE
PATH = r"D:\\Google Drive\\University\\IML\\IML Ex 2\\"  # todo: remove!


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
    # irrelevant_columns = ['id', 'date', 'price', 'long', 'lat']
    # weak_correlation_columns = ['yr_renovated', 'sqft_lot', 'sqft_lot15',
    #                             'condition', 'yr_built', 'zipcode', 'long']
    # x_df = X.dropna().drop_duplicates()  # drop rows with NA values
    #
    # positive_columns = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'yr_built', 'zipcode']
    # x_df = x_df[(x_df[positive_columns] > 0).all(axis=1)]
    #
    # x_df = x_df.loc[x_df['waterfront'].isin(range(2)) &
    #                 x_df['view'].isin(range(5)) &
    #                 x_df['condition'].isin(range(1, 6)) &
    #                 x_df['grade'].isin(range(1, 15)) &
    #                 x_df['bedrooms'].isin(range(1, 12))]
    #
    # x_df = x_df.drop(columns=irrelevant_columns + weak_correlation_columns)  # drop columns
    # return x_df if y is None else x_df, y.loc[y.index.isin(x_df.index)]
    x_df = X.drop(columns=['date', 'id', 'price']).dropna().drop_duplicates()
    positive_columns = ['yr_built', 'sqft_living', 'sqft_lot', 'bathrooms', 'bedrooms', 'floors',
                        'sqft_above', 'sqft_living15', 'sqft_lot15', 'sqft_basement',
                        'yr_renovated']
    x_df = x_df[(x_df[positive_columns] > 0).all(axis=1)]

    categorical_features = ['view', 'waterfront', 'condition', 'grade',
                            'yr_built', 'yr_renovated', 'zipcode', 'bedrooms']
    x_df[categorical_features] = x_df[categorical_features].astype(int)
    x_df = x_df.loc[x_df['waterfront'].isin(range(2)) &
                    x_df['view'].isin(range(5)) &
                    x_df['condition'].isin(range(1, 6)) &
                    x_df['grade'].isin(range(1, 15)) &
                    x_df['bedrooms'].isin(range(1, 13))]

    low_correlation_columns = ['condition', 'long', 'lat', 'zipcode', 'sqft_lot', 'sqft_lot15',
                               'yr_built', 'yr_renovated']
    x_df = x_df.drop(columns=low_correlation_columns)

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
    y_std = y.std()
    if y_std == 0:
        return
    # lst = [] # todo: remove
    for feature_name in X.columns:
        feature = X[feature_name]
        f_std = feature.std()
        if f_std == 0:
            continue
        coef = np.round(feature.cov(y) / (f_std * y_std), 3)
        # lst.append((feature_name, coef)) # todo: remove
        px.scatter(x=feature, y=y, trendline="ols", trendline_color_override='black',
                   title=f"{coef}-Pearson Correlation between {feature_name} Values and Price",
                   labels={"x": f"{feature_name} values", "y": "Price"}) \
            .write_image(output_path + f"/{feature_name}_correlation.png")
    # for x in sorted(lst, key=lambda x: x[1], reverse=True): # todo: remove
    #     print(x)
    # quit()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    raw_train_X, raw_train_y, raw_test_X, raw_test_y = split_train_test(df, df[['price']])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(raw_train_X, raw_train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, PATH)  # todo: remove path

    # Question 4 - Fit model over increasing percentages of the overall training data
    m_samples = 10
    percentage = np.arange(10, 101)
    out = np.empty((percentage.shape[0], m_samples))
    test_X, test_y = preprocess_data(raw_test_X, raw_test_y)
    for i, f in enumerate(percentage / 100):
        for j in range(m_samples):
            X_sample = train_X.sample(frac=f)
            y_sample = train_y.sample(frac=f)
            out[i, j] = LinearRegression().fit(X_sample, y_sample).loss(test_X, test_y)
    mean = out.mean(axis=1)
    std = out.std(axis=1)
    px.scatter(x=percentage, y=mean, title="MSE as a Function of Training Set Size",
               labels={"x": "Training Set Size in Percentage", "y": "MSE Values"}) \
        .add_scatter(x=percentage, y=mean - 2 * std, mode="lines",
                     line=dict(color="lightgrey"), showlegend=False) \
        .add_scatter(x=percentage, y=mean + 2 * std, mode="lines",
                     line=dict(color="lightgrey"), showlegend=False, fill='tonexty') \
        .write_image(PATH + r"Q4Linear.png")
