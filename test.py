import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdata
from datetime import date
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
import plotly.express as px
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start="2015-01-01"
today=date.today().strftime("%Y-%m-%d")

st.title("Stock Trend Analysis")

stocks=("AAPL","GOOG","MSFT","GME")
st.header("Select Stock Ticker for Analysis")
selected_stocks=st.selectbox("Select",stocks)

data_load_state=st.text("Loading data...")
data=yf.download(selected_stocks,start,today)
data=data.drop(['Adj Close'],axis=1)

st.subheader("Raw data")
st.write(data.tail())
st.write(data.describe().drop(['count'],axis=0))
data_load_state.text("Loading data...done!")

data=data.reset_index()

st.subheader("Time Series Data")
fig=go.Figure();
fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Stock Open'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='Stock Close'))
fig.update_layout(xaxis_rangeslider_visible=True,template='plotly_dark')
st.plotly_chart(fig)

st.subheader("Time Series Data with Candlestick")
figure = go.Figure(data=[go.Candlestick(x=data['Date'],open=data['Open'],high=data['High'],low=data['Low'],close=data['Close'])])
figure.update_layout(xaxis_rangeslider_visible=True,height=600)
st.plotly_chart(figure)

st.subheader('Stock closing price with Time Period Selector')
fig1 = px.line(data,x='Date',y='Close')
fig1.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=3, label='3m', step='month', stepmode='backward'),
            dict(count=6, label='6m',step='month', stepmode='backward'),
            dict(count=1, label='1y',step='year', stepmode='backward'),
            dict(step='all')
        ])
    )
)
fig1.update_traces(line_color='#e52b50')
st.plotly_chart(fig1)

ma100=data.Close.rolling(100).mean()
ma200=data.Close.rolling(200).mean()

st.subheader("Closing Price vs Time Chart with 100 days Moving Average")
fig=plt.figure(figsize=(10,5))
plt.plot(data.Close)
plt.plot(ma100,'r')
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100 days Moving Average and 200 days Moving Average")
fig=plt.figure(figsize=(10,5))
plt.plot(data.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)

data_training=pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

model=load_model('keras_model.h5')
past_100_days=data_training.tail(100)
final_data=[past_100_days,data_testing]
final_data=pd.concat(final_data)
input_data=scaler.fit_transform(final_data)
x_test=[];
y_test=[];

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader("Original Closing Price vs Predicted Closing Price")
fig2=plt.figure(figsize=(10,5))
plt.plot(y_test,'black',label='Original Price')
plt.plot(y_predicted,'green', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
