import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

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
        .write_image(PATH + "TempDay.png")

    month_df = df.groupby(['Month'])['Temp'].std().reset_index()
    px.bar(month_df, x=month_df.index, y='Temp',
           title='Standard Deviation of Daily Temperature as a Function of Month') \
        .write_image(PATH + "MonthTempDev.png")

    # Question 3 - Exploring differences between countries
    countries_df = df.groupby(['Country', 'Month']).\
        agg(mean=('Temp', 'mean'), std=('Temp', 'std')).reset_index()
    px.line(countries_df, x='Month', y='mean', color='Country', error_y='std') \
        .write_image(PATH + "AverageCountryMonthTemp.png")

# Returning to the full dataset, group the samples according to `Country` and `Month` and
# calculate the average and standard deviation of the temperature. Plot a line plot of the average
# monthly temperature, with error bars (using the standard deviation) color coded by the country.
# If using plotly.express.line have a look at the #todo error_y argument

# Question 4 - Fitting model for different values of `k`
# raise NotImplementedError()

# Question 5 - Evaluating fitted model on different countries
# raise NotImplementedError()
