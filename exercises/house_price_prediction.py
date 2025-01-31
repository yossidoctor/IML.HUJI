from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


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
    cleaned_df = X.dropna()
    cleaned_df = remove_irrelevant_features(cleaned_df)
    cleaned_df = validate_positive_values(cleaned_df)
    cleaned_df = validate_non_negative_features(cleaned_df)
    cleaned_df = validate_categorical_features(cleaned_df)
    cleaned_df = remove_sqft_outliers(cleaned_df)
    cleaned_df = remove_low_correlation_features(cleaned_df)
    if 'price' in df.columns:
        cleaned_df = cleaned_df.drop(columns=['price'])
    cleaned_df = cleaned_df.drop_duplicates()
    if y is None:
        return cleaned_df, None
    return cleaned_df, y.loc[y.index.isin(cleaned_df.index)]


def remove_low_correlation_features(cleaned_df):
    low_correlation_features = ['condition', 'sqft_lot', 'sqft_lot15', 'yr_built', 'yr_renovated']
    return cleaned_df.drop(columns=low_correlation_features)


def remove_sqft_outliers(cleaned_df):
    sqft_outliers_upper_boundary = 0.99
    sqft_columns = cleaned_df.filter(like='sqft').columns
    upper_quantiles = cleaned_df[sqft_columns].quantile(sqft_outliers_upper_boundary)
    is_not_outlier_upper = (cleaned_df[sqft_columns] <= upper_quantiles).all(axis=1)
    return cleaned_df[is_not_outlier_upper]


def validate_categorical_features(cleaned_df):
    categorical_features = ['waterfront', 'view', 'condition', 'grade', 'bedrooms']
    cleaned_df[categorical_features] = cleaned_df[categorical_features].astype(int)
    return cleaned_df.loc[cleaned_df['waterfront'].isin(range(2)) &
                          cleaned_df['view'].isin(range(5)) &
                          cleaned_df['condition'].isin(range(1, 6)) &
                          cleaned_df['grade'].isin(range(1, 13)) &
                          cleaned_df['bedrooms'].isin(range(1, 12))]


def validate_non_negative_features(cleaned_df):
    non_negative_features = ['sqft_basement', 'yr_renovated', 'bathrooms', 'bedrooms', 'floors',
                             'sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
    return cleaned_df[(cleaned_df[non_negative_features] >= 0).all(axis=1)]


def validate_positive_values(cleaned_df):
    positive_features = ['yr_built']
    if 'price' in df.columns:
        positive_features += ['price']
    return cleaned_df[(cleaned_df[positive_features] > 0).all(axis=1)]


def remove_irrelevant_features(cleaned_df):
    irrelevant_features = ['date', 'id', 'long', 'lat', 'zipcode']
    return cleaned_df.drop(columns=irrelevant_features)


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
    for feature_name in X.columns:
        feature = X[feature_name]
        f_std = feature.std()
        if f_std == 0:
            continue
        coefficient = np.round(feature.cov(y) / (f_std * y_std), 3)
        title = f"{coefficient}-Pearson Correlation between {feature_name} Values and Price"
        labels = {"x": f"{feature_name} values", "y": "Price"}
        image_path = output_path + f"/{feature_name}_correlation.png"
        px.line(x=feature, y=y, title=title, labels=labels).write_image(image_path)


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    raw_train_X, raw_train_y, raw_test_X, raw_test_y = split_train_test(df, df[['price']])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(raw_train_X, raw_train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

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
        .write_image(r"Q4Linear.png")
