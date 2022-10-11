# Import Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error
import datetime
from datetime import date
from keras.models import load_model
import yfinance as yf
from plotly import graph_objs as go

import warnings
warnings.filterwarnings('ignore')

# Input Variables

user_input = st.sidebar.text_input('Enter Stock name','TATAMOTORS.NS')
st.title('Stock Price Prediction')
period = st.sidebar.number_input('Number of days of Forecast', value = int(10))
data_range = st.sidebar.number_input('Number of Previous Days', value= int(int(period) * 2.5))
data_range = int(data_range)

start=st.sidebar.date_input('Start Date',value=datetime.datetime(2012, 7, 15))
end=st.sidebar.date_input('End Date',value= datetime.datetime(2022,7,31))

# Importing Data for 10 years
ticker = yf.Ticker(user_input)
data = ticker.history(start= start, end= end)
data = data.reset_index()

#Displaying Data
st.subheader('Date from 2012-2022')
st.write(data.describe())

# Visualization
st.subheader('Closing Price vs Time ')
# fig = plt.figure(figsize= (22,8))
# plt.plot(data.Close, 'r', label = 'Closing Prices')
# plt.legend(loc = 'upper left')
# st.pyplot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(x= data.Date, y= data.Close))
fig.layout.update(xaxis_rangeslider_visible= True, width= 850, height= 500)
st.plotly_chart(fig)

# Preprocessing and Partition of data 
dataset = data.filter(['Close']).values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

training_size = int(len(dataset) * 0.75)
test_data = scaled_data[training_size - 90:, :]

x_test, y_test = [], dataset[training_size:, :]
for j in range(90, len(test_data)):
    x_test.append(test_data[j - 90:j, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Calling Model
model = load_model('stock_prediction.h5')


# Prediction on testing data

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

Train = data[:training_size]
Validation = data[training_size:]
Validation['Prediction'] = predictions

# Actual vs Predicted Prices
st.subheader('Train & Test Vs Predicted ')
fig2 = plt.figure(figsize = (22,8))
Train.Close.plot(label= 'Train');
Validation.Close.plot(label = 'Test');
Validation.Prediction.plot(label = 'Predicted');
plt.legend(loc= 'upper left');
st.pyplot(fig2)

# Forecasting Future Prices

initial_data = [i for i in data['Close'].values]


for i in np.arange(int(period)):
    in_array = np.array(initial_data[-90:]).reshape(-1,1)
    scaled_in = scaler.fit_transform(in_array)
    values = np.array(scaled_in).reshape(1,-1)
    values = values.reshape(values.shape[0], values.shape[1], 1)
    pred = model.predict(values)
    inverse = scaler.inverse_transform(pred)
    initial_data.extend(inverse[0].tolist())

st.subheader('Predicted Prices')
st.table(initial_data[-int(period):])

# Visualizing Forecasted Values

st.subheader('Test data & Forecasted Values')
fig3 = plt.figure(figsize= (18,6))
plt.plot()
plt.plot(range(data_range), data.Close[-data_range:], label= 'Actual');
plt.plot(range(data_range, data_range + int(period)), initial_data[-int(period):], label = 'Forecasted');
plt.legend(loc= 'upper left');
st.pyplot(fig3)


