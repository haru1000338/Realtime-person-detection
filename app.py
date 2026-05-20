import streamlit as st
import pandas as pd
import plotly.express as px
import time

# ページの設定
st.set_page_config(page_title="オープンキャンパス動線分析", layout="wide")
st.title("📊 リアルタイム滞在分析ダッシュボード")

# データの読み込み
def load_data():
    try:
        # csvの読み込み（列名を指定）
        df = pd.read_csv("dwell_log.csv", names=["Timestamp", "Booth_name", "Track_ID", "Dwell_Time_sec"])
        # 1行目がヘッダー文字列だった場合は除外する
        if df.iloc[0]['Timestamp'] == 'Timestamp':
            df = df.iloc[1:]
        
        # 時間を数値に変換
        df['Dwell_Time_sec'] = df['Dwell_Time_sec'].astype(float)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# データが存在する場合の表示処理
if not df.empty:
    # 🌟 上段：サマリー情報
    col1, col2 = st.columns(2)
    
    total_visitors = df['Track_ID'].nunique()
    col1.metric("総検知人数", f"{total_visitors} 人")
    
    total_dwell = df['Dwell_Time_sec'].sum()
    col2.metric("総滞在時間", f"{total_dwell:.1f} 秒")

    st.markdown("---")

    # 🌟 中段：グラフ表示
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("ブース別の訪問者数")
        booth_counts = df.groupby('Booth_name')['Track_ID'].nunique().reset_index()
        fig_bar = px.bar(booth_counts, x='Booth_name', y='Track_ID', color='Booth_name', text_auto=True)
        st.plotly_chart(fig_bar, width='stretch')

    with col4:
        st.subheader("ブース別の平均滞在時間（秒）")
        avg_dwell = df.groupby('Booth_name')['Dwell_Time_sec'].mean().reset_index()
        fig_pie = px.pie(avg_dwell, values='Dwell_Time_sec', names='Booth_name', hole=0.4)
        st.plotly_chart(fig_pie, width='stretch')

    # 🌟 下段：生データ
    st.markdown("---")
    st.subheader("📝 最新の生ログデータ")
    st.dataframe(df.tail(10).sort_index(ascending=False))

else:
    st.warning("データが見つかりません。カメラシステムを起動してCSVを生成してください。")

# 自動更新ボタン
if st.button("🔄 データを最新に更新"):
    st.rerun()

auto_refresh = st.checkbox("自動更新（2秒ごと）", value=True)
if auto_refresh:
    time.sleep(2)
    st.rerun()