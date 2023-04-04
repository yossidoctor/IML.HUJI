import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"

PATH = r"D:\\Google Drive\\University\\IML\\IML Ex 2\\"  # todo: remove!


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['DayOfYear'] = pd.to_datetime(df['Date'], format='%D/%M/%Y').dt.dayofyear
    df = df[df['Temp'] > -12]  # removing invalid data
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df["Country"] == "Israel"]
    px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year',
               title="Temperature as a Function of Day of Year in Israel") \
        .write_image(PATH + "Q2PolyFit_1.png")

    month_df = df.groupby(['Month'])['Temp'].std().reset_index()
    px.bar(month_df, x=month_df.index, y='Temp',
           title='Standard Deviation of Daily Temperature as a Function of Month') \
        .write_image(PATH + "Q2PolyFit_2.png")

    # Question 3 - Exploring differences between countries
    countries_df = df.groupby(['Country', 'Month']). \
        agg(mean=('Temp', 'mean'), std=('Temp', 'std')).reset_index()
    px.line(countries_df, x='Month', y='mean', color='Country', error_y='std') \
        .write_image(PATH + "Q3PolyFit.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df['DayOfYear'], df['Temp'])
    k_values = np.arange(10)
    loss_q4 = [round(PolynomialFitting(k + 1).fit(train_X, train_y).loss(test_X, test_y), 2)
               for k in k_values]
    print(loss_q4)
    px.bar(x=k_values, y=loss_q4, title="Test Error as a function of Polynomial Degree") \
        .write_image(PATH + "Q4PolyFit.png")

    # Question 5 - Evaluating fitted model on different countries
    best_k = 5  # TODO: VALIDATE
    israel_fit = PolynomialFitting(best_k).fit(israel_df['DayOfYear'], israel_df['Temp'])
    loss_q5 = []
    for i, country in enumerate(df[df["Country"] != "Israel"]["Country"].unique()):
        country_df = df.loc[df["Country"] == country]
        loss_q5[i] = round(israel_fit.loss(country_df['DayOfYear'], country_df['Temp']), 2)
    px.bar(x=k_values, y=loss_q5, title="Test Error as a function of Polynomial Degree") \
        .write_image(PATH + "Q5PolyFit.png")
