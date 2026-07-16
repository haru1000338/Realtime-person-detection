import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os

# ページの設定
st.set_page_config(page_title="オープンキャンパス動線分析", layout="wide", initial_sidebar_state="expanded")
st.title("📊 リアルタイム滞在分析ダッシュボード")

# 🌟 データの読み込み
@st.cache_data(ttl=5)
def load_data(filepath="dwell_log.csv"):
    if not os.path.exists(filepath):
        return pd.DataFrame()

    column_names = ["Timestamp", "Booth_name", "Track_ID", "Real_ID", "Status", "ReID_Score", "Dwell_Time_sec"]
    try:
        df = pd.read_csv(filepath, names=column_names, dtype=str)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if df.iloc[0]['Timestamp'] == 'Timestamp':
        df = df.iloc[1:]

    df["Dwell_Time_sec"] = pd.to_numeric(df["Dwell_Time_sec"], errors='coerce').fillna(0.0)
    df["ReID_Score"] = pd.to_numeric(df["ReID_Score"], errors='coerce').fillna(0.0)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.dropna(subset=["Timestamp"])
    
    # 🌟 ポイント1: waiting（判定待ちのゴミデータ）の完全除去は維持
    df = df[df["Status"] != "waiting"]
    
    return df

df = load_data()

if not df.empty:
    # 🌟 サイドバー：フィルター機能
    st.sidebar.header("🔍 フィルター設定")
    target = st.sidebar.radio(
        "分析対象を選択", 
        ["来場者（高校生）のみ", "全員（スタッフ含む）", "スタッフのみ"]
    )

    if target == "来場者（高校生）のみ":
        filtered_df = df[df["Status"] != "staff"]
    elif target == "スタッフのみ":
        filtered_df = df[df["Status"] == "staff"]
    else:
        filtered_df = df

    if filtered_df.empty:
        st.info("選択された条件に一致するデータが現在ありません。")
    else:
        # ==========================================
        # 画面描画スタート
        # ==========================================
        col1, col2 = st.columns(2)

        # 1. 総来場者数（ユニークなReal_IDの数）
        visitors_count = filtered_df['Real_ID'].nunique()
        col1.metric("総来場者数", f"{visitors_count} 人")

        # 🌟 ポイント3: 「1人あたりの平均カメラ内滞在時間」
        # まず人(Real_ID)ごとに合計滞在時間を算出し、その平均を取る
        avg_per_person = filtered_df.groupby('Real_ID')['Dwell_Time_sec'].sum().mean()
        col2.metric("1人あたりの平均カメラ内滞在時間", f"{avg_per_person:.1f} 秒")

        st.markdown("---")

        st.subheader(f"🏢 ブース別分析（対象: {target}）")
        chart_col1, chart_col2 = st.columns(2)

        # 🌟 ポイント2: 人ベースでの合算をやめ、生のセッション（ログ1行ごと）を尊重する
        # ブース別訪問者数（そのブースに訪れたユニークな人数）
        booth_visitors = filtered_df.groupby('Booth_name')['Real_ID'].nunique().reset_index(name='訪問者数')
        
        # ブース別の滞在時間データ（1回の訪問セッションあたりの平均・最大）
        booth_times = filtered_df.groupby('Booth_name').agg(
            平均滞在時間_秒=('Dwell_Time_sec', 'mean'),
            最大滞在時間_秒=('Dwell_Time_sec', 'max')
        ).reset_index()

        # データを結合
        booth_stats = pd.merge(booth_visitors, booth_times, on='Booth_name')

        with chart_col1:
            # 訪問者数の棒グラフ
            fig_bar = px.bar(
                booth_stats, x='Booth_name', y='訪問者数', 
                color='Booth_name', text_auto=True,
                title="ブース別 訪問者数",
                labels={'訪問者数': '訪問者数 (人)', 'Booth_name': 'ブース名'}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")

        with chart_col2:
            st.markdown("**📊 ブース別 滞在時間データ (1セッションあたり)**")
            
            # 見やすいように丸める
            booth_stats['平均滞在時間_秒'] = booth_stats['平均滞在時間_秒'].round(1)
            booth_stats['最大滞在時間_秒'] = booth_stats['最大滞在時間_秒'].round(1)
            booth_stats = booth_stats.rename(columns={'Booth_name': 'ブース名'})
            
            st.dataframe(booth_stats, width="stretch", hide_index=True)

        # 🌟 下段：生データプレビュー
        st.markdown("---")
        st.subheader("📝 最新の生ログデータ (ノイズ除去済)")
        display_df = filtered_df.copy().sort_index(ascending=False)
        st.dataframe(
            display_df.head(20),
            width="stretch",
            column_config={
                "ReID_Score": st.column_config.NumberColumn("AI類似度", format="%.2f"),
                "Dwell_Time_sec": st.column_config.NumberColumn("滞在時間(秒)", format="%.1f"),
                "Status": st.column_config.TextColumn("ステータス")
            }
        )

else:
    st.warning("データが見つかりません。カメラシステムを起動してCSVを生成してください。")

# 🔄 自動更新機能
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("自動更新（2秒ごと）", value=True)
if auto_refresh:
    time.sleep(2)
    st.rerun()

if st.sidebar.button("🔄 手動で最新に更新"):
    st.rerun()