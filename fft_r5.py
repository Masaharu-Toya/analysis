import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image
import numpy as np
import pandas as pd

primaryColor="#e6728f"
backgroundColor="#140909"
secondaryBackgroundColor="#02061c"
textColor="#f3f380"
font="sans serif"


st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='auto') 
stc.html("<p style='color:#00ff00;font-size:32pt;'> 周波数解析アプリ </p>")   
##st.file_uploader("ファイルアップロード", type='csv')
uploaded_file = st.sidebar.file_uploader("１．ファイルアップをロード", type='csv') 
if uploaded_file is None:
    col1,col2=st.beta_columns(2)
    #col1.write('操作説明')
    # col1.write('１．ファイルをアップロード')  
    # col1.write('２．分析する項目を選択')
    # col2.write('ファイルが指定されていません')
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # カラム選択
    column = st.sidebar.selectbox(
        '２．分析する項目を選択',
        (df.columns.values))
    st.sidebar.write('分析する項目:', column)

# 選択肢
filter=st.sidebar.radio('３．ノイズフィルタ選択',('ローパス', 'ハイパス', 'バンドパス','バンドエリミネーション'))
# 数値入力

if uploaded_file is None:
    Num=st.sidebar.number_input('４．サンプリング数入力')
   
else:
    Data_num=df[column]
    Num_d=len(Data_num)
    Num=st.sidebar.number_input('４．サンプリング数入力', 0, 999999,Num_d)
T=st.sidebar.number_input('５．サンプリング周期(ms)を入力')

if uploaded_file is None:
    col2.write(text10)

elif uploaded_file is not None:
    # アップロードファイルをメイン画面にデータ表示
    col2.write('アップロードファイルのデータ')
    col2.write(df)
    st.sidebar.write('８．実行ボタンをクリック')
    if st.sidebar.button('実行'):
        if T==0:
            st.sidebar.write('サンプリング周期が入力されていません')
        else:
            import numpy as np
            import pandas as pd

            # データのパラメータ
            N = Num            # サンプル数
            dt = T/1000          # 周波数軸
            fc1 = fc11  # カットオフ周波数1[Hz]
            fc2 = fc22  # カットオフ周波数2[Hz]
            #A1, A2 = 20, 5
            t = np.arange(0, N*dt, dt)   # 時間軸
            freq = np.linspace(0, 1.0/dt, N)  # 周波数軸

            f = df[column]
            # 高速フーリエ変換（周波数信号に変換）
            F = np.fft.fft(f)

        # 正規化 + 交流成分2倍
            F = F / (N / 2)

        # 直流成分は等倍に戻す
            F[0] = F[0]/2

            

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio

            fig = make_subplots(rows=2, cols=2,subplot_titles=['元データのトレンドグラフ','元データのFFT結果','フィルタ後のトレンドグラフ','フィルタ後のFFT結果'])

            fig.add_trace(
                go.Scatter(x=t, y=f,marker={'color':'red'}),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=freq, y=np.abs(F),marker={'color':'red'}),
                row=1, col=2
            )
            fig.update_layout(height=600, width=1300, title_text="トレンドグラフとFFT結果",showlegend=False)

            st.plotly_chart(fig, use_container_width=True)







fc11=st.sidebar.number_input('６．カットオフ周波数 fc1(Hz)を入力', 0, 999999)
fc22=st.sidebar.number_input('７．カットオフ周波数 fc2(Hz)を入力 ', 0, 999999)


text1='操作説明'
text2='１．ファイルをアップロード'  
text3='２．分析する項目を選択'
text4='３．ノイズフィルタを選択'
text5='４．サンプリング数を入力'
text6='５．サンプリング周期を入力'
text7='６．カットオフ周波数をfc1を入力'
text8='７．カットオフ周波数をfc2を入力'
text9='８．実行ボタンをクリック'
text10='ファイルが指定されていません'

col1,col2=st.beta_columns(2)
col1.write(text1)
col1.write(text2)  
col1.write(text3)
col1.write(text4)
col1.write(text5)
col1.write(text6)
col1.write(text7)
col1.write(text8)
col1.write(text9)

if uploaded_file is None:
    col2.write(text10)

elif uploaded_file is not None:
    # アップロードファイルをメイン画面にデータ表示
    col2.write('アップロードファイルのデータ')
    col2.write(df)
    st.sidebar.write('８．実行ボタンをクリック')
    if st.sidebar.button('実行'):
        if T==0:
            st.sidebar.write('サンプリング周期が入力されていません')
        else:
            import numpy as np
            import pandas as pd

            # データのパラメータ
            N = Num            # サンプル数
            dt = T/1000          # 周波数軸
            fc1 = fc11  # カットオフ周波数1[Hz]
            fc2 = fc22  # カットオフ周波数2[Hz]
            #A1, A2 = 20, 5
            t = np.arange(0, N*dt, dt)   # 時間軸
            freq = np.linspace(0, 1.0/dt, N)  # 周波数軸

            f = df[column]
            # 高速フーリエ変換（周波数信号に変換）
            F = np.fft.fft(f)

        # 正規化 + 交流成分2倍
            F = F / (N / 2)

        # 直流成分は等倍に戻す
            F[0] = F[0]/2

            # 配列Fをコピー
            F2 = F.copy()

            if filter=='バンドパス':
                
            # バンドパスフィルタ処理（fc1～fc2[Hz]の帯域以外を0にする）
                F2[(freq > fc2)] = 0
                F2[(freq < fc1)] = 0
            elif filter=='バンドエリミネーション' :
            # バンドストップ処理（fc1～fc2[Hz]の帯域を0にする）
                F2[((freq > fc1)&(freq < fc2))] = 0 + 0
            elif filter=='ローパス' :        
            # ローパスフィルタ処理（fc1超を0にする）
                F2[(freq > fc1)] = 0
            elif filter=='ハイパス' :
            # ハイパスフィルタ処理（fc1未満を0にする）
                F2[(freq < fc1)] = 0

            # 高速逆フーリエ変換（時間信号に戻す）
            f2 = np.fft.ifft(F2)

            # 振幅を元のスケールに戻す
            f2 = np.real(f2*N)

            # 周波数軸のデータを保存
            spectrum_df = pd.DataFrame({})
            spectrum_df["freq"] = freq
            spectrum_df.set_index("freq", inplace=True)
            spectrum_df["F1"] = np.abs(F)
            spectrum_df["F2"] = np.abs(F2)
            spectrum_df.to_csv(r"C:\Users\touya\spectrum_" + str(fc1) + "_" + str(fc2) + ".csv")

        # 時間軸のデータを保存
            amplitude_df = pd.DataFrame({})
            amplitude_df["time"] = t
            #amplitude_df.set_index("time", inplace=True)
            amplitude_df["f"] = f
            amplitude_df["f2"] = f2
            amplitude_df.to_csv(r"C:\Users\touya\amplitude_" + str(fc1) + "_" + str(fc2) + ".csv")

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
            import plotly.io as pio

            fig = make_subplots(rows=2, cols=2,subplot_titles=['元データのトレンドグラフ','元データのFFT結果','フィルタ後のトレンドグラフ','フィルタ後のFFT結果'])

            fig.add_trace(
                go.Scatter(x=t, y=f,marker={'color':'red'}),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=freq, y=np.abs(F),marker={'color':'red'}),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=t, y=f2,marker={'color':'#00ffff'}),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=freq, y=np.abs(F2),marker={'color':'#00ffff'}),
                row=2, col=2
            )
            fig.update_layout(height=600, width=1300, title_text="トレンドグラフとFFT結果",showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            fig2=go.Figure()
            fig2.add_trace(
                go.Scatter(x=t,y=f,mode = 'lines',name = '元データ',marker={'color':'red'})
            )
            fig2.add_trace(
                go.Scatter(x=t,y=f2,mode = 'lines',name = 'フィルタ後',marker={'color':'#00ffff'})
            )
            fig2.update_layout(height=500, width=1300, title_text="トレンドグラフ重ね合わせ比較")

            st.plotly_chart(fig2)
            #stc.iframe("http://localhost/covid19_predict.html",height=500, width=1300)

stc.html(
    """<header style='color:#afeeee;font-weight:bold;'> <h3>ノイズフィルタの概要<br>
    <p style='color:#cc6633;font-size:11pt;font-weight:bold;'>
    ローパスフィルタ  ：fc1超を0にする。<br>
    ハイパスフィルタ：fc1未満を0にする。<br>
    バンドパスフィルタ：fc1～fc2[Hz]の帯域以外を0にする。<br>
    バンドエリミネーションフィルタ：fc1～fc2[Hz]の帯域を0にする"""
    )