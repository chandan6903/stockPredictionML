import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model

import streamlit as st

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", "AAPL")

api_key = "MSUXUGI52VP00ZLX"

stock_symbol = user_input
start_date = '2000-01-01'
end_date = '2022-12-31'

ts = TimeSeries(key=api_key, output_format='pandas')

data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')

df = data[(data.index >= start_date) & (data.index <=end_date)]

#describing data

st.subheader("Data From 2000-2022")
st.write(df.describe())

#visualization

st.subheader("Closing Price Vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df['4. close'])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 days moving average.")
ma100 = df['4. close'].rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df['4. close'])
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100 days and 200 days moving average")
ma100 = df["4. close"].rolling(100).mean()
ma200 = df["4. close"].rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df["4. close"])
st.pyplot(fig)

#splitting Data into Training and Testing

data_training = pd.DataFrame(df["4. close"][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df["4. close"][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load my model

model = load_model("my_model1.keras")

past_100_Days = data_training.tail(100)
final_df = past_100_Days._append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = (1/scaler[0])
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph

st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, "b", label = "Original Price")
plt.plot(y_predicted, "r", label= "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

