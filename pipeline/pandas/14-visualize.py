#!/usr/bin/env python3
"""
    script to visualize the pd.DataFrame:

    - The column Weighted_Price should be removed
    - Rename the column Timestamp to Date
    - Convert the timestamp values to date values
    - Index the data frame on Date
    - Missing values in Close should be set to the
    previous row value
    - Missing values in High, Low, Open should be set
    to the same row’s Close value
    - Missing values in Volume_(BTC) and Volume_(Currency)
    should be set to 0
    - Plot the data from 2017 and beyond at daily intervals
    and group the values of the same day such that:
        - High: max
        - Low: min
        - Open: mean
        - Close: mean
        - Volume(BTC): sum
        - Volume(Currency): sum
"""

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file(
    "../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")

# Remove the Weighted_Price column
df = df.drop(columns="Weighted_Price")

# Rename the column Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert the timestamp values to datetime values
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index the data frame on Date
df = df.set_index("Date")

# Ensure all index values are of datetime type
df.index = pd.to_datetime(df.index)

# Missing values in Close should be set to the previous row value
df["Close"] = df["Close"].fillna(method="ffill")

# Missing values in High, Low, Open should be set to the same row’s Close value
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Resample and aggregate the data
resampled_df = df["2017":].resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

# Plot the data
resampled_df.plot()
plt.show()
