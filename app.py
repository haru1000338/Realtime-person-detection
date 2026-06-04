import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
from datetime import datetime

# ==========================================
# 🌟 ページ全体の設定
# ==========================================
st.set_page_config(page_title="AIリアルタイム動線分析", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stMetric { background-color: #2b2b2b; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    /* caption（小さな説明文）を見やすくするための調整 */
    .st-emotion-cache-16idsys p { font-size: 0.85rem; color: #aaaaaa; margin-top: -10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 📊 データ読み込み・前処理関数
# ==========================================
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
    df = df[df["Status"] != "waiting"]
    
    return df

df = load_data()

# ==========================================
# 🎛️ サイドバー（操作パネル）
# ==========================================
with st.sidebar:
    st.header("⚙️ コントロールパネル")
    
    target = st.radio(
        "👥 分析対象の切り替え", 
        ["来場者（高校生）のみ", "全員（スタッフ含む）"]
    )
    
    st.divider()
    auto_refresh = st.checkbox("🔄 2秒ごとに自動更新", value=True)
    
    if st.button("🔄 手動で最新に更新"):
        st.rerun()

# ==========================================
# 🌟 メインダッシュボード領域
# ==========================================
st.title("📊 オープンキャンパス リアルタイム滞在分析")

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<p style='text-align: right; color: gray;'>最終データ取得: {current_time}</p>", unsafe_allow_html=True)

# --- データフィルタリング ---
if not df.empty:
    if target == "来場者（高校生）のみ":
        filtered_df = df[df["Status"] != "staff"]
    else:
        filtered_df = df
else:
    filtered_df = pd.DataFrame()

if filtered_df.empty:
    st.warning("⚠️ 現在、指定された条件のデータがありません。AIの推論を開始してください。")
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    st.stop() 

# ==========================================
# 🎯 2. サマリー（ビッグナンバーと計算式）
# ==========================================
col1, col2, col3 = st.columns(3)

visitors_count = filtered_df['Real_ID'].nunique() if 'Real_ID' in filtered_df.columns else 0
avg_stay_per_person = filtered_df.groupby('Real_ID')['Dwell_Time_sec'].sum().mean() if 'Real_ID' in filtered_df.columns else 0
avg_reid_score = filtered_df['ReID_Score'].mean() if 'ReID_Score' in filtered_df.columns else 0

with col1:
    st.metric(label="👥 総来場者数", value=f"{visitors_count} 人")
    st.caption("※ 算出基準: AIが特定した重複のない人物IDの総数")

with col2:
    val_stay = avg_stay_per_person if pd.notna(avg_stay_per_person) else 0.0
    st.metric(label="⏱️ 1人あたりの平均滞在時間", value=f"{val_stay:.1f} 秒")
    st.caption("※ 算出基準: ブースへの合計滞在秒数 ÷ 総来場者数")

with col3:
    val_score = avg_reid_score * 100 if pd.notna(avg_reid_score) else 0.0
    st.metric(label="🤖 平均AI確信度 (Re-ID)", value=f"{val_score:.1f} %")
    st.caption("※ 算出基準: 全ログの類似度スコアの平均（システムの安定度）")

st.divider()

# ==========================================
# 🏢 3. ブース別分析グラフ（表 ＆ 円グラフ）
# ==========================================
if 'Booth_name' in filtered_df.columns:
    st.subheader(f"📈 ブース別 パフォーマンス（対象: {target}）")
    col_chart1, col_chart2 = st.columns(2)

    booth_stats = filtered_df.groupby('Booth_name').agg(
        訪問者数=('Real_ID', 'nunique'),
        合計時間_秒=('Dwell_Time_sec', 'sum'),
        平均時間_秒=('Dwell_Time_sec', 'mean')
    ).reset_index()
    
    booth_stats['合計時間_秒'] = booth_stats['合計時間_秒'].round(1)
    booth_stats['平均時間_秒'] = booth_stats['平均時間_秒'].round(1)
    booth_stats = booth_stats.rename(columns={'Booth_name': 'ブース名'})

    with col_chart1:
        st.markdown("**📊 ブース別 集計データ**")
        st.dataframe(booth_stats, use_container_width=True, hide_index=True)

    with col_chart2:
        booth_sum = filtered_df.groupby('Booth_name')['Dwell_Time_sec'].sum().reset_index()
        if not booth_sum.empty:
            fig_pie = px.pie(
                booth_sum, names='Booth_name', values='Dwell_Time_sec', 
                title="🍩 会場全体の滞在時間シェア",
                hole=0.4
            )
            # 円グラフの計算根拠を小さく追加
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("※ シェアの計算式: 各ブースの合計滞在時間 ÷ 全ブースの合計滞在時間")

st.divider()

# ==========================================
# 📋 4. 生データプレビュー（色付けなし）
# ==========================================
st.subheader("📋 最新の生ログデータ (計算の根拠)")
st.write("※ ここに表示されているデータをもとに上のグラフや数値を計算しています。")

if not filtered_df.empty:
    display_df = filtered_df.copy().sort_index(ascending=False).head(50)
    
    st.dataframe(
        display_df, 
        use_container_width=True, 
        height=300,
        column_config={
            "ReID_Score": st.column_config.NumberColumn("AI確信度", format="%.2f"),
            "Dwell_Time_sec": st.column_config.NumberColumn("滞在時間(秒)", format="%.1f")
        }
    )

# ==========================================
# 🔄 画面の一番最後で自動更新をトリガーする
# ==========================================
if auto_refresh:
    time.sleep(2)
    st.rerun()