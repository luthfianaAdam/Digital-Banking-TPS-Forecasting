import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
# from future_predict import future_predict
import datetime

df = pd.read_csv('result.csv')
df1 = pd.read_csv('transaction-datasets.csv')
df1 = df1[['tanggal','jml_transaksi']].rename(columns={'jml_transaksi': 'ribu_transaksi'})
df1['tanggal'] = pd.to_datetime(df1['tanggal'])
df1['tanggal'] = df1['tanggal'].dt.strftime('%Y/%m')

background = open("textfile/background.txt", "r")
insight = open("textfile/insight.txt", "r")
datasets = open("textfile/datasets.txt", "r")
author = open("textfile/author.txt", "r")
caption = open("textfile/caption.txt", "r")

st.title("Digital Banking TPS Forecasting")

st.header('Background')
st.subheader("Jumlah Transaksi Bank Indonesia (BI)")
st.bar_chart(data=df1, x="tanggal", y="ribu_transaksi")
st.markdown(background.read())


st.subheader("LSTM Model")
line_chart = alt.Chart(df).mark_line().encode(
x='tanggal:T',
y='tps:Q',
color='data:N'
)
st.altair_chart(line_chart, use_container_width=True)

st.caption(caption.read())

# Set the Forecasting parameter
inputfile = 'transaction-datasets.csv'
modelname = 'future_predict_models/model1.keras'
predict_range = st.slider(
    'Proyeksi Jumlah Bulan Selanjutnya',
    min_value=0,
    max_value=24,
    step=1,
    value=12
)
result = future_predict(inputfile,modelname,predict_range)
result = result.rename(columns={'tps': 'Train', 'predict': 'Prediction'})
# st.write(result)
result = result.melt('tanggal', var_name='data', value_name='tps')
# st.write(result)

st.subheader("TPS Forecasting")
line_chart = alt.Chart(result).mark_line().encode(
x='tanggal:T',
y='tps:Q',
color='data:N'
)
st.altair_chart(line_chart, use_container_width=True)



st.subheader('Insight')
st.markdown(insight.read())

st.subheader('source')
st.markdown(datasets.read())

st.subheader('Author')
st.markdown(author.read())
