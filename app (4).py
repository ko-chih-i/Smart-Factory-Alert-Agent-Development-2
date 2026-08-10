"""
智慧工廠設備異常警報儀表板 (Smart Factory Equipment Anomaly Alert Dashboard)
檔案名稱: app.py
套件需求: streamlit, plotly, pandas, numpy, scikit-learn

執行方式:
  pip install -r requirements.txt
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="Pegatron Smart Factory Anomaly Alert Dashboard",
    page_icon="🏭",
    layout="wide"
)

# 注入淺藍色主題客製化 CSS
st.markdown("""
    <style>
    /* 全域主背景與文字顏色調整為淺藍風格 */
    .stApp {
        background-color: #f0f8ff; /* 淺愛麗絲藍 */
    }
    .stSidebar {
        background-color: #e6f2ff; /* 側邊欄稍深淺藍 */
    }
    /* 卡片與容器樣式 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #bae6fd;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Cache Synthetic Generator
@st.cache_data
def generate_sensor_data(num_rows=200, anomaly_ratio=0.12, inject_missing=False, missing_rate=0.04, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    start_time = datetime.strptime("2024-06-03 19:05:00", "%Y-%m-%d %H:%M:%S")
    data = []

    for i in range(num_rows):
        current_time = (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        is_abnormal = random.random() < anomaly_ratio

        if is_abnormal:
            anomaly_type = random.choice(['high_temp', 'low_temp', 'high_press', 'low_press', 'high_vib', 'compound'])
            temp = round(random.uniform(45.0, 50.0), 1)
            pressure = round(random.uniform(1.00, 1.05), 2)
            vibration = round(random.uniform(0.02, 0.04), 2)

            if anomaly_type == 'high_temp':
                temp = round(random.uniform(52.1, 64.0), 1)
            elif anomaly_type == 'low_temp':
                temp = round(random.uniform(34.0, 42.5), 1)
            elif anomaly_type == 'high_press':
                pressure = round(random.uniform(1.09, 1.35), 2)
            elif anomaly_type == 'low_press':
                pressure = round(random.uniform(0.80, 0.96), 2)
            elif anomaly_type == 'high_vib':
                vibration = round(random.uniform(0.08, 0.18), 2)
            elif anomaly_type == 'compound':
                temp = round(random.uniform(53.0, 62.0), 1)
                pressure = round(random.uniform(1.10, 1.25), 2)
                vibration = round(random.uniform(0.08, 0.15), 2)
            
            label = 'abnormal'
        else:
            temp = round(random.uniform(45.0, 50.0), 1)
            pressure = round(random.uniform(1.00, 1.05), 2)
            vibration = round(random.uniform(0.02, 0.04), 2)
            label = 'normal'

        # Optional Missing Value Injection (~4% rate)
        if inject_missing and random.random() < missing_rate:
            target_col = random.choice(['temp', 'pressure', 'vibration'])
            if target_col == 'temp':
                temp = None
            elif target_col == 'pressure':
                pressure = None
            else:
                vibration = None

        data.append({
            'timestamp': current_time,
            'temp': temp,
            'pressure': pressure,
            'vibration': vibration,
            'label': label
        })

    return pd.DataFrame(data)

# Pipeline Function
def run_anomaly_pipeline(df, impute_method="線性插值 (Linear Interpolation)"):
    clean_df = df.copy()
    imputed_info = []

    for col in ['temp', 'pressure', 'vibration']:
        missing_count = clean_df[col].isnull().sum()
        if missing_count > 0:
            if "線性插值" in impute_method or "Linear" in impute_method:
                clean_df[col] = clean_df[col].interpolate(method='linear').bfill().ffill()
                method_name = "線性插值 (Linear)"
            elif "前向填補" in impute_method or "LOCF" in impute_method or "Forward" in impute_method:
                clean_df[col] = clean_df[col].ffill().bfill()
                method_name = "前向填補 (Forward Fill / LOCF)"
            else:
                median_val = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median_val)
                method_name = f"中位數 ({median_val:.2f})"

            imputed_info.append(f"{col}: 填補 {missing_count} 筆 [{method_name}]")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df[['temp', 'pressure', 'vibration']])
    clean_df['temp_z'], clean_df['pressure_z'], clean_df['vibration_z'] = scaled[:, 0], scaled[:, 1], scaled[:, 2]

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(clean_df[['temp', 'pressure', 'vibration']])
    iso_preds = iso_forest.predict(clean_df[['temp', 'pressure', 'vibration']])
    iso_scores = iso_forest.decision_function(clean_df[['temp', 'pressure', 'vibration']])

    reasons_list, warning_reasons, severities, anomaly_scores, actions, predicted_labels = [], [], [], [], [], []

    for idx, row in clean_df.iterrows():
        reasons = []
        score = 0.0

        if row['temp'] > 52.0:
            reasons.append(f"過熱 ({row['temp']}°C > 52°C)")
            score += 0.45
        elif row['temp'] < 43.0:
            reasons.append(f"過冷 ({row['temp']}°C < 43°C)")
            score += 0.40

        if row['pressure'] > 1.08:
            reasons.append(f"壓力過高 ({row['pressure']} > 1.08 bar)")
            score += 0.40
        elif row['pressure'] < 0.97:
            reasons.append(f"壓力過低 ({row['pressure']} < 0.97 bar)")
            score += 0.35

        if row['vibration'] > 0.07:
            reasons.append(f"劇烈震動 ({row['vibration']} > 0.07 g)")
            score += 0.50

        is_abnormal = (len(reasons) > 0) or (iso_preds[idx] == -1)
        final_score = round(min(1.0, max(score, float(0.5 - iso_scores[idx]))), 2)
        
        if is_abnormal:
            pred_label = 'abnormal'
            sev = 'CRITICAL' if final_score >= 0.75 else ('HIGH' if final_score >= 0.50 else 'WARNING')
            if "過熱" in str(reasons):
                act = "檢查冷卻泵浦與水管流量，降低機台負載 25%"
            elif "震動" in str(reasons):
                act = "安排主軸軸承雷射對心，補給潤滑油脂"
            elif "壓力" in str(reasons):
                act = "檢修氣壓歧管與分流閥，確認有無漏氣"
            else:
                act = "派員進行感測器校正與基礎巡檢"
            warn_reason = f"Isolation Forest 檢測到特徵偏離邊界 (Score={final_score:.2f})。屬早期警告，提示維護巡檢。"
        else:
            pred_label, sev, act = 'normal', 'NORMAL', '設備運作正常，維持預防性維護'
            warn_reason = '感測器數值在標準公差範圍內 (Normal)'

        reasons_list.append(", ".join(reasons) if reasons else ("孤立森林離群" if is_abnormal else "正常"))
        warning_reasons.append(warn_reason)
        severities.append(sev)
        anomaly_scores.append(final_score)
        actions.append(act)
        predicted_labels.append(pred_label)

    clean_df['predicted_label'] = predicted_labels
    clean_df['severity'] = severities
    clean_df['anomaly_score'] = anomaly_scores
    clean_df['warning_reason'] = warning_reasons
    clean_df['root_cause'] = reasons_list
    clean_df['action_suggestion'] = actions
    if 'label' in clean_df.columns:
        clean_df['gt_match'] = (clean_df['predicted_label'] == clean_df['label']).map({True: '✓ 一致 (Match)', False: '✗ 差異 (Diff)'})
    return clean_df, imputed_info

def main():
    st.sidebar.title("和碩 Pegatron 智慧工廠")
    st.sidebar.caption("Assignment 3 — Streamlit + Plotly 警報儀表板")

    num_rows = st.sidebar.slider("感測器筆數 (Rows)", 100, 500, 200, 50)
    anomaly_prob = st.sidebar.slider("異常機率", 0.05, 0.30, 0.12, 0.01)
    inject_missing = st.sidebar.toggle("可選遺失值處理 (Missing Values Imputation)", value=True, help="注入約 4% 感測器遺失值 (NaN) 並由 AI Agent 執行自動補值處理")

    impute_method = "線性插值 (Linear Interpolation)"
    if inject_missing:
        impute_method = st.sidebar.selectbox(
            "填補演算法 (Imputation Strategy)",
            options=[
                "線性插值 (Linear Interpolation)",
                "前向填補 / LOCF (Forward Fill)",
                "中位數填補 (Median Imputation)"
            ],
            index=0
        )

    uploaded_file = st.sidebar.file_uploader("匯入感測器 CSV 檔案", type=["csv"])

    sample_csv_template = """timestamp,temp,pressure,vibration,label
2024-06-03 19:05:00,46.2,1.02,0.03,normal
2024-06-03 19:06:00,47.1,1.01,0.02,normal
2024-06-03 19:07:00,58.4,1.12,0.09,abnormal
2024-06-03 19:08:00,45.8,1.03,0.03,normal
2024-06-03 19:09:00,39.2,0.91,0.02,abnormal
2024-06-03 19:10:00,48.0,1.04,0.04,normal"""

    st.sidebar.download_button(
        label="📄 下載 CSV 建議上傳範例 (.csv)",
        data=sample_csv_template,
        file_name="sample_sensor_upload.csv",
        mime="text/csv"
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = generate_sensor_data(num_rows=num_rows, anomaly_ratio=anomaly_prob, inject_missing=inject_missing)

    processed_df, imputed_info = run_anomaly_pipeline(raw_df, impute_method=impute_method)

    if imputed_info:
        msg_lines = ["🧹 自動數據清洗補值報告:"] + [f"• {info}" for info in imputed_info]
        st.sidebar.info("\n".join(msg_lines))

    st.title("🏭 智慧工廠設備異常警報儀表板")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("🧮 1. 怎麼算出 Anomaly Score？（分數算式說明）", expanded=False):
            st.markdown("""
            **Anomaly Score 推導與計算機制**：
            1. **Z-Score 特徵標準化**：Z = (X - μ) / σ
            2. **Isolation Forest 孤立樹離群評估**：計算決策分數 S_raw。
            3. **分數歸一化 (0% ~ 100%)**：Anomaly Score = Clip((0.5 - S_raw) * 100%, 0%, 100%)
            """)

    with col_exp2:
        with st.expander("📁 2. 上傳 CSV 建議格式規範與範例說明", expanded=False):
            st.markdown("""
            **上傳 CSV 建議欄位名稱與資料型別**：
            - 'timestamp'、'temp' (°C)、'pressure' (bar)、'vibration' (g)、'label' (normal/abnormal)
            """)

    total_count = len(processed_df)
    abnormal_df = processed_df[processed_df['predicted_label'] == 'abnormal']
    abnormal_count = len(abnormal_df)

    gt_matches = (processed_df['predicted_label'] == processed_df['label']).sum() if 'label' in processed_df.columns else total_count
    accuracy = (gt_matches / total_count * 100) if total_count > 0 else 100.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("總筆數", f"{total_count} 筆")
    m2.metric("GT 比對正確率", f"{accuracy:.1f}%", delta=f"{gt_matches}/{total_count} Match")
    m3.metric("正常狀態", f"{total_count - abnormal_count} 筆")
    m4.metric("異常警報", f"{abnormal_count} 筆", delta=f"{abnormal_count/total_count*100:.1f}%", delta_color="inverse")
    m5.metric("緊急警報", f"{len(processed_df[processed_df['severity']=='CRITICAL'])} 筆", delta_color="inverse")

    st.markdown("---")

    # ---------------------------------------------------------
    # 修改重點：Plotly 淺藍色系與高對比折線顏色設定
    # ---------------------------------------------------------
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=processed_df['timestamp'], y=processed_df['temp'], 
        mode='lines', name='溫度 (°C)', line=dict(color='#d97706', width=2) # 深橘色
    ))
    fig_ts.add_trace(go.Scatter(
        x=processed_df['timestamp'], y=processed_df['pressure'], 
        mode='lines', name='壓力 (bar)', line=dict(color='#0284c7', width=2), yaxis='y2' # 湛藍色
    ))
    fig_ts.add_trace(go.Scatter(
        x=processed_df['timestamp'], y=processed_df['vibration'], 
        mode='lines', name='震動 (g)', line=dict(color='#7c3aed', width=2), yaxis='y3' # 深紫色
    ))

    if abnormal_count > 0:
        fig_ts.add_trace(go.Scatter(
            x=abnormal_df['timestamp'], y=abnormal_df['temp'], 
            mode='markers', name='異常點', marker=dict(color='#dc2626', size=10, symbol='x') # 深紅色標記
        ))

    # 改為淺藍色背景底色 (`#f0f9ff` & `#e0f2fe`) 與深灰色文字 (`#1e293b`)
    fig_ts.update_layout(
        paper_bgcolor='#f0f9ff',  # 外部邊框顏色：淺蔚藍
        plot_bgcolor='#e0f2fe',   # 圖表畫布背景：柔和淺藍
        font=dict(color='#1e293b', size=12), # 文字顏色：深灰（高可讀性）
        height=450,
        hovermode="x unified", 
        legend=dict(orientation="h", y=1.15, x=0, bgcolor='rgba(255,255,255,0.6)'),
        yaxis=dict(
            title=dict(text="溫度 (°C)", font=dict(color='#d97706')),
            gridcolor='#bae6fd', # 網格線改為淺藍色
            zerolinecolor='#7dd3fc'
        ), 
        yaxis2=dict(
            title=dict(text="壓力 (bar)", font=dict(color='#0284c7')),
            overlaying='y', side='right',
            gridcolor='#bae6fd',
            zerolinecolor='#7dd3fc'
        ),
        yaxis3=dict(
            title=dict(text="震動 (g)", font=dict(color='#7c3aed')),
            overlaying='y', side='right', position=0.95,
            gridcolor='#bae6fd',
            zerolinecolor='#7dd3fc'
        )
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("🚨 異常警報紀錄與 Gemini AI 建議處置")
    st.dataframe(
        processed_df[['timestamp', 'temp', 'pressure', 'vibration', 'severity', 'anomaly_score', 'root_cause', 'action_suggestion']],
        use_container_width=True
    )

if __name__ == "__main__":
    main()
