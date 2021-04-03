import pandas as pd
import datetime
import math
import streamlit as st
import streamlit.components.v1 as stc

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='auto') 

st.sidebar.write('COVID19 Google感染予測')

query ='SELECT * FROM `bigquery-public-data.covid19_public_forecasts.japan_prefecture_28d`'

data_frame = pd.read_gbq(query, 'friendly-lamp-299810')

data_frame["num"]= data_frame.prefecture_code.str.extract(r'(\d+)').astype(int)

df=data_frame.groupby("prediction_date").sum().reset_index()

df["prefecture_code"]="JP-00"
df["num"]=0
df["prefecture_name_kanji"]="日本全体"
df["prefecture_name"]="Japan_total"
df.loc[df['new_confirmed'] == 0, 'new_confirmed'] = None
df.loc[df['cumulative_confirmed'] == 0, 'cumulative_confirmed'] = None

df1=data_frame.append(df)
df2=df1.sort_values(by=["num"], ascending=True)
df2["select_code"]=df2["prefecture_code"] + df2["prefecture_name_kanji"]

df2.to_csv("covid19.csv",encoding="shift-jis")

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
dfs = list(df2.groupby("select_code"))

first_title = dfs[0][0][5:] + "の新型コロナ感染者数予測"

traces = []
buttons = []

colors = {
    'background': '#111111',
    'text': 'red'
}




#print(first_title[5:])

for i,d in enumerate(dfs):
    visible = [False] * len(dfs)
    visible[i] = True
    name = d[0]
    traces.append(
        px.bar(d[1],
               x="prediction_date",
               y="new_confirmed_ground_truth",
               color="new_confirmed_ground_truth").update_traces(visible=True if i==0 else False).data[0],           
    )
   
   
for i,e in enumerate(dfs):
    visible = [False] * len(dfs)
    visible[i] = True
    name = e[0]
   
    traces.append(
        px.scatter(e[1],
               x="prediction_date",
               y="new_confirmed",
               color="new_confirmed").update_traces(visible=True if i==0 else False).data[0],           
    )
    
        
    buttons.append(dict(label=name[5:],
                        method="update",
                        args=[{"visible":visible},
                              {"title":f"{name[5:] }" +"の新型コロナ感染者数予測"}]))

updatemenus = [{'active':0, "buttons":buttons}]
fig = go.Figure(data=traces,
                 layout=dict(updatemenus=updatemenus))
fig.update_layout(
    title=first_title,
    title_x=0.5,
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    width = 1200,       # 全体のサイズ
    height = 750,
    autosize = True   # HTMLで表示したときページに合わせてリサイズするかどうか
         
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
st.plotly_chart(fig)