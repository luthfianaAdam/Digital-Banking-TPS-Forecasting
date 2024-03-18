import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime

st.set_page_config(layout='wide')

train_result = pd.read_csv('train_result.csv')
val_result = pd.read_csv('val_result.csv')
test_result = pd.read_csv('test_result.csv')
hariraya = pd.read_csv('hari_raya.csv') 
df1 = pd.read_csv('transaction-datasets2.csv') 
df1 = df1.rename(columns={'overall_transaksi': 'Overall Transaction (Ribu)','phone_banking':'Phone Banking Transaction (Ribu)', 'sms_mobile_banking':'SMS/Mobile Banking Transaction (Ribu)', 'internet_banking':'Internet Banking Transaction (Ribu)', 'tanggal': 'Tanggal'})
df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
df1['Tanggal'] = df1['Tanggal'].dt.strftime('%Y-%m')

background_caption = open("textfile/background.txt", "r")
insight = open("textfile/insight.txt", "r")
datasets = open("textfile/datasets.txt", "r")
author = open("textfile/author.txt", "r")
caption = [open("textfile/caption_datasets.txt", "r"), open("textfile/hariraya.txt", "r"), open("textfile/caption.txt", "r")]
overview = [open("textfile/overview_dataset.txt", "r"), open("textfile/overview_seasonality.txt", "r"), open("textfile/overview_model.txt", "r")]

# Remove delta arrow from st.metric
st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Digital Banking Transaction Forecasting (2019 - 2023)")

# =====================================BACKGROUND=====================================
st.header('Background')

st.markdown(background_caption.read())

st.subheader("Jumlah Transaksi Digital Banking Bank Indonesia (BI)")

 # Transaction details
pb, mb, ib = df1['Phone Banking Transaction (Ribu)'].sum(), df1['SMS/Mobile Banking Transaction (Ribu)'].sum(), df1['Internet Banking Transaction (Ribu)'].sum()
total = pb+mb+ib

# Transaction type details
pb_byr, pb_intra, pb_antar = df1['pb_pembayaran'].sum(), df1['pb_intrabank'].sum(), df1['pb_antarbank'].sum()
pb_total = pb_byr+pb_intra+pb_antar
mb_byr, mb_intra,  mb_antar= df1['mb_pembayaran'].sum(), df1['mb_intrabank'].sum(), df1['mb_antarbank'].sum()
mb_total = mb_byr+mb_intra+mb_antar
ib_byr, ib_intra, ib_antar = df1['ib_pembayaran'].sum(), df1['ib_intrabank'].sum(), df1['ib_antarbank'].sum()
ib_total = ib_byr+ib_intra+ib_antar

overall_byr = pb_byr+mb_byr+ib_byr
overall_intra = pb_intra+mb_intra+ib_intra
overall_antar = pb_antar+mb_antar+ib_antar
overall_total = pb_total+mb_total+ib_total

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overall Transaction','Overall Transaction type','Phone Banking','SMS/Mobile Banking','Internet Banking'])
with tab1:
    # Modify dataframe to visualize in stacked bar chart
    overall = df1[["Tanggal", "Phone Banking Transaction (Ribu)","SMS/Mobile Banking Transaction (Ribu)","Internet Banking Transaction (Ribu)"]]
    overall = overall.rename(columns={'Phone Banking Transaction (Ribu)':'Phone Banking','SMS/Mobile Banking Transaction (Ribu)':'SMS/Mobile Banking','Internet Banking Transaction (Ribu)':'Internet Banking'})
    overall = pd.melt(overall, id_vars=['Tanggal'], value_vars=["Phone Banking","SMS/Mobile Banking","Internet Banking"])
    overall = overall.rename(columns={'variable':'Transaction Method', 'value':'Transaction (Ribu)'})
    
    col1, col2 = st.columns([7,1])
    with col1:
        st.bar_chart(data=overall, x="Tanggal", y="Transaction (Ribu)", color='Transaction Method')
    with col2:
        st.metric("Phone Banking", value=f'{pb/1000:.2f}Jt', delta=f'{100*pb/total:.2f}%', delta_color="off")
        st.metric("SMS/Mobile Banking", value=f'{mb/1000000:.2f}M', delta=f'{100*mb/total:.2f}%', delta_color="off")
        st.metric("Internet Banking", value=f'{ib/1000000:.2f}M', delta=f'{100*ib/total:.2f}%', delta_color="off")
with tab2:
    overall_type = df1[["Tanggal"]]
    overall_type['Pembayaran/Pembelian'] = df1[['pb_pembayaran','mb_pembayaran','ib_pembayaran']].sum(axis=1).tolist()
    overall_type['Intrabank'] = df1[['pb_intrabank','mb_intrabank','ib_intrabank']].sum(axis=1)
    overall_type['Antarbank'] = df1[['pb_antarbank','mb_antarbank','ib_antarbank']].sum(axis=1)
    temp = np.zeros(len(df1))
    temp[:31] = df1.loc[:30,'Overall Transaction (Ribu)'].tolist()
    overall_type['None'] = temp
    overall_type = pd.melt(overall_type, id_vars=['Tanggal'], value_vars=["None","Pembayaran/Pembelian","Intrabank","Antarbank"])
    overall_type = overall_type.rename(columns={'variable':'Transaction Type', 'value':'Transaction (Ribu)'})
    col1, col2 = st.columns([7,1])
    with col1:    
        st.bar_chart(data=overall_type, x="Tanggal", y="Transaction (Ribu)", color='Transaction Type')
    with col2:
        st.metric("Pembayaran/Pembelian", value=f'{overall_byr/1000000:.2f}M', delta=f'{100*overall_byr/overall_total:.2f}%', delta_color="off")
        st.metric("Intrabank", value=f'{overall_intra/1000000:.2f}M', delta=f'{100*overall_intra/overall_total:.2f}%', delta_color="off")
        st.metric("Antarbank", value=f'{overall_antar/1000000:.2f}M', delta=f'{100*overall_antar/overall_total:.2f}%', delta_color="off")
with tab3:
    phone_banking = df1[["Tanggal","pb_pembayaran","pb_intrabank","pb_antarbank"]]
    phone_banking = phone_banking.rename(columns={'pb_pembayaran':'Pembayaran/Pembelian','pb_intrabank':'Intrabank','pb_antarbank':'Antarbank'})
    temp = np.zeros(len(df1))
    temp[:31] = df1.loc[:30,'Phone Banking Transaction (Ribu)'].tolist()
    phone_banking['None'] = temp
    phone_banking = pd.melt(phone_banking, id_vars=['Tanggal'], value_vars=["None","Pembayaran/Pembelian","Intrabank","Antarbank"])
    phone_banking = phone_banking.rename(columns={'variable':'Transaction Type', 'value':'Transaction (Ribu)'})

    col1, col2 = st.columns([7,1])
    with col1:    
        st.bar_chart(data=phone_banking, x="Tanggal", y="Transaction (Ribu)", color='Transaction Type')
    with col2:
        st.metric("Pembayaran/Pembelian", value=f'{pb_byr:.2f}Rb', delta=f'{100*pb_byr/pb_total:.2f}%', delta_color="off")
        st.metric("Intrabank", value=f'{pb_intra:.2f}Rb', delta=f'{100*pb_intra/pb_total:.2f}%', delta_color="off")
        st.metric("Antarbank", value=f'{pb_antar:.2f}Rb', delta=f'{100*pb_antar/pb_total:.2f}%', delta_color="off")

with tab4:
    mobile_banking = df1[["Tanggal","mb_pembayaran","mb_intrabank","mb_antarbank"]]
    mobile_banking = mobile_banking.rename(columns={'mb_pembayaran':'Pembayaran/Pembelian','mb_intrabank':'Intrabank','mb_antarbank':'Antarbank'})
    temp = np.zeros(len(df1))
    temp[:31] = df1.loc[:30,'SMS/Mobile Banking Transaction (Ribu)'].tolist()
    mobile_banking['None'] = temp
    mobile_banking = pd.melt(mobile_banking, id_vars=['Tanggal'], value_vars=["None","Pembayaran/Pembelian","Intrabank","Antarbank"])
    mobile_banking = mobile_banking.rename(columns={'variable':'Transaction Type', 'value':'Transaction (Ribu)'})

    col1, col2 = st.columns([7,1])
    with col1:
        st.bar_chart(data=mobile_banking, x="Tanggal", y="Transaction (Ribu)", color='Transaction Type')
    with col2:
        st.metric("Pembayaran/Pembelian", value=f'{mb_byr/1000000:.2f}M', delta=f'{100*mb_byr/mb_total:.2f}%', delta_color="off")
        st.metric("Intrabank", value=f'{mb_intra/1000000:.2f}M', delta=f'{100*mb_intra/mb_total:.2f}%', delta_color="off")
        st.metric("Antarbank", value=f'{mb_antar/1000000:.2f}M', delta=f'{100*mb_antar/mb_total:.2f}%', delta_color="off")

with tab5:
    internet_banking = df1[["Tanggal","ib_pembayaran","ib_intrabank","ib_antarbank"]]
    internet_banking = internet_banking.rename(columns={'ib_pembayaran':'Pembayaran/Pembelian','ib_intrabank':'Intrabank','ib_antarbank':'Antarbank'})
    temp = np.zeros(len(df1))
    temp[:31] = df1.loc[:30,'Internet Banking Transaction (Ribu)'].tolist()
    internet_banking['None'] = temp
    internet_banking = pd.melt(internet_banking, id_vars=['Tanggal'], value_vars=["None","Pembayaran/Pembelian","Intrabank","Antarbank"])
    internet_banking = internet_banking.rename(columns={'variable':'Transaction Type', 'value':'Transaction (Ribu)'})
    col1, col2 = st.columns([7,1])
    with col1:
        st.bar_chart(data=internet_banking, x="Tanggal", y="Transaction (Ribu)", color='Transaction Type')
    with col2:
        st.metric("Pembayaran/Pembelian", value=f'{ib_byr/1000000:.2f}M', delta=f'{100*ib_byr/ib_total:.2f}%', delta_color="off")
        st.metric("Intrabank", value=f'{ib_intra/1000000:.2f}M', delta=f'{100*ib_intra/ib_total:.2f}%', delta_color="off")
        st.metric("Antarbank", value=f'{ib_antar/1000000:.2f}M', delta=f'{100*ib_antar/ib_total:.2f}%', delta_color="off")

st.caption(caption[0].read())
st.markdown(overview[0].read())

# =====================================Seasonality=====================================
st.subheader("Seasonality")
hariraya['tanggal'] = pd.to_datetime(hariraya['tanggal'])
temp_hariraya = hariraya
hariraya['tanggal'] = hariraya['tanggal'].dt.strftime('%Y-%m')
lebaran = hariraya.loc[hariraya['hari_raya'] == 'idul fitri']
lebaran = lebaran['tanggal'].tolist()
imlek = hariraya.loc[hariraya['hari_raya'] == 'imlek']
imlek = imlek['tanggal'].tolist()
df1['hari_raya'] = None
df1.loc[df1['Tanggal'].isin(lebaran),'hari_raya'] = 'idul fitri'
df1.loc[df1['Tanggal'].isin(imlek),'hari_raya'] = 'imlek'
# df1.loc[df1['Tanggal'][-3:] == '-12','hari_raya'] = 'akhir tahun'
column1 = ['Phone Banking Transaction (Ribu)', 'SMS/Mobile Banking Transaction (Ribu)', 'Internet Banking Transaction (Ribu)']
# kuartal dataframe
kuartal = pd.DataFrame(columns=['Q', 'Phone Banking Transaction (Ribu)', 'SMS/Mobile Banking Transaction (Ribu)', 'Internet Banking Transaction (Ribu)', 'Year'])
Q = []
year = []
for x in ['2019','2020','2021','2022','2023']:
    for y in range(4):
        Q.append(f'{x}-Q{y+1}')
        year.append(x)
kuartal['Q'] = Q
kuartal['Year'] = year
for i in column1:
    kuartal[i] = df1[i].groupby(df1.index // 3).sum()
kuartal['Overall'] = kuartal[column1].sum(axis=1)
# print(column1)
# print(kuartal[column1].sum(axis=1))

tabA, tabB =st.tabs(['Kuartal', 'Hari Raya'])
with tabA:
    tabA1, tabA2, tabA3, tabA4 = st.tabs(['Phone Banking','SMS/Mobile Banking','Internet Banking', 'Overall'])
    with tabA1:
        st.bar_chart(data=kuartal, x="Q", y="Phone Banking Transaction (Ribu)", color='Year')
    with tabA2:
        st.bar_chart(data=kuartal, x="Q", y="SMS/Mobile Banking Transaction (Ribu)", color='Year')
    with tabA3:
        st.bar_chart(data=kuartal, x="Q", y="Internet Banking Transaction (Ribu)", color='Year')
    with tabA4:
        st.bar_chart(data=kuartal, x="Q", y="Overall", color='Year')
with tabB:
    tabB1, tabB2, tabB3, tabB4 = st.tabs(['Phone Banking','SMS/Mobile Banking','Internet Banking', 'Overall'])
    with tabB1:
        st.bar_chart(data=df1, x="Tanggal", y="Phone Banking Transaction (Ribu)", color="hari_raya")
    with tabB2:
        st.bar_chart(data=df1, x="Tanggal", y="SMS/Mobile Banking Transaction (Ribu)", color="hari_raya")
    with tabB3:
        st.bar_chart(data=df1, x="Tanggal", y="Internet Banking Transaction (Ribu)", color="hari_raya")
    with tabB4:
        st.bar_chart(data=df1, x="Tanggal", y="Overall Transaction (Ribu)", color="hari_raya")


st.caption(caption[1].read())
st.markdown(overview[1].read())

# =====================================FORECASTING MODEL=====================================
st.subheader("LSTM Model")
tabC, tabD, tabE =st.tabs(['Training', 'Validation', 'Testing'])
with tabC:
    train_result = pd.melt(train_result, id_vars=['tanggal'], value_vars=['Train Predictions','Actuals'])
    train_result = train_result.rename(columns={'variable':'Data', 'value':'TPS', 'tanggal': 'Tanggal'})
    # print(train_result)
    line_chart = alt.Chart(train_result).mark_line().encode(
    x='Tanggal',
    y='TPS:Q',
    color='Data:N'
    )
    st.altair_chart(line_chart, use_container_width=True)

with tabD:
    val_result = pd.melt(val_result, id_vars=['tanggal'], value_vars=['Validation Predictions','Actuals'])
    val_result = val_result.rename(columns={'variable':'Data', 'value':'TPS', 'tanggal': 'Tanggal'})
    # print(val_result)
    line_chart = alt.Chart(val_result).mark_line().encode(
    x='Tanggal',
    y='TPS:Q',
    color='Data:N'
    )
    st.altair_chart(line_chart, use_container_width=True)

with tabE:
    test_result = pd.melt(test_result, id_vars=['tanggal'], value_vars=['Test Predictions','Actuals'])
    test_result = test_result.rename(columns={'variable':'Data', 'value':'TPS', 'tanggal': 'Tanggal'})
    # print(test_result)
    print(test_result)
    line_chart = alt.Chart(test_result).mark_line().encode(
    x='Tanggal',
    y='TPS:Q',
    color='Data:N'
    )
    st.altair_chart(line_chart, use_container_width=True)

# st.caption(caption[2].read())
st.markdown(overview[2].read())

col1, col2 = st.columns([5,2])
with col1:
    # =====================================INSIGHT=====================================
    st.subheader('Conclusion')
    st.markdown(insight.read())
with col2:
    # =====================================SOURCE=====================================
    st.subheader('source')
    st.markdown(datasets.read())
    # =====================================AUTHOR=====================================
    st.subheader('Author')
    st.markdown(author.read())

st.subheader('Next Improvement')
st.markdown(open("textfile/next_improvement.txt", "r").read())
