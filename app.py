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

    # 【修正箇所】総滞在時間を全体の平均滞在時間（mean）に置き換え
    avg_dwell = df['Dwell_Time_sec'].mean()
    col2.metric("全体の平均滞在時間", f"{avg_dwell:.1f} 秒")

    # ブースごとの平均滞在時間を計算して表示
    avg_by_booth = df.groupby('Booth_name')['Dwell_Time_sec'].mean().reset_index()
    avg_by_booth['Dwell_Time_sec'] = avg_by_booth['Dwell_Time_sec'].round(1)

    st.subheader("ブース別平均滞在時間（秒）")
    avg_col1, avg_col2 = st.columns([2, 1])
    with avg_col1:
        fig_avg = px.bar(avg_by_booth, x='Booth_name', y='Dwell_Time_sec', color='Booth_name', text='Dwell_Time_sec')
        fig_avg.update_layout(showlegend=False, yaxis_title='平均滞在時間 (秒)')
        st.plotly_chart(fig_avg, use_container_width=True)
    with avg_col2:
        st.table(avg_by_booth.rename(columns={'Dwell_Time_sec': '平均滞在時間 (秒)'}))

    st.markdown("---")

    # 🌟 中段：グラフ表示
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("ブース別の訪問者数")
        booth_counts = df.groupby('Booth_name')['Track_ID'].nunique().reset_index()
        fig_bar = px.bar(booth_counts, x='Booth_name', y='Track_ID', color='Booth_name', text_auto=True)
        st.plotly_chart(fig_bar, width='stretch')

    with col4:
        st.subheader("ブース別の滞在時間分布（秒）")
        # 箱ひげ図で滞在時間の分布（中央値、四分位、外れ値）を表示
        fig_box = px.box(df, x='Booth_name', y='Dwell_Time_sec', points='outliers', color='Booth_name')
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

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

with st.expander("システム詳細(クリックして展開)"):
    st.markdown("""
                システムの処理概要
                1. カメラから映像を取得
                2. AIモデル(YOLOv26)で人を検出し、足の位置を特定
                3. 足の位置をもとにヒートマップを生成
                4. 人の位置と滞在時間をCSVに記録
                5. StreamlitでCSVを読み込み、ダッシュボードを表示
                6. ダッシュボードは総検知人数、全体の平均滞在時間、ブース別訪問者数、ブース別平均滞在時間を表示
                7. データは自動で更新され、最新の状態を反映
                """)