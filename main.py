import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

df = pd.read_csv('result.csv')
background = open("background.txt", "r")
insight = open("insight.txt", "r")
datasets = open("datasets.txt", "r")
author = open("author.txt", "r")
caption = open("caption.txt", "r")

st.title("Digital Banking TPS Forecasting")

st.header('Background')
st.markdown(background.read())


st.subheader("TPS Forecasting")
line_chart = alt.Chart(df).mark_line().encode(
x='tanggal:T',
y='tps:Q',
color='data:N'
)
st.altair_chart(line_chart, use_container_width=True)

st.caption(caption.read())

st.subheader('Insight')
st.markdown(insight.read())

st.subheader('source')
st.markdown(datasets.read())

st.subheader('Author')
st.markdown(author.read())