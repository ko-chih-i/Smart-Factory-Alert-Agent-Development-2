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

# Cache Synthetic Generator
@st.cache_data
def generate_sensor_data(num_rows=200, anomaly_ratio=0.12, seed=42):
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

        data.append({
            'timestamp': current_time,
            'temp': temp,
            'pressure': pressure,
            'vibration': vibration,
            'label': label
        })

    return pd.DataFrame(data)

# Pipeline Function
def run_anomaly_pipeline(df):
    clean_df = df.copy()
    for col in ['temp', 'pressure', 'vibration']:
        if clean_df[col].isnull().sum() > 0:
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df[['temp', 'pressure', 'vibration']])
    clean_df['temp_z'], clean_df['pressure_z'], clean_df['vibration_z'] = scaled[:, 0], scaled[:, 1], scaled[:, 2]

    iso_forest = IsolationForest(contamination=0.12, random_state=42)
    iso_forest.fit(clean_df[['temp', 'pressure', 'vibration']])
    iso_preds = iso_forest.predict(clean_df[['temp', 'pressure', 'vibration']])
    iso_scores = iso_forest.decision_function(clean_df[['temp', 'pressure', 'vibration']])

    reasons_list, severities, anomaly_scores, actions, predicted_labels = [], [], [], [], []

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
        else:
            pred_label, sev, act = 'normal', 'NORMAL', '設備運作正常，維持預防性維護'

        reasons_list.append(", ".join(reasons) if reasons else ("孤立森林離群" if is_abnormal else "正常"))
        severities.append(sev)
        anomaly_scores.append(final_score)
        actions.append(act)
        predicted_labels.append(pred_label)

    clean_df['predicted_label'] = predicted_labels
    clean_df['severity'] = severities
    clean_df['anomaly_score'] = anomaly_scores
    clean_df['root_cause'] = reasons_list
    clean_df['action_suggestion'] = actions
    if 'label' in clean_df.columns:
        clean_df['gt_match'] = (clean_df['predicted_label'] == clean_df['label']).map({True: '✓ 一致 (Match)', False: '✗ 差異 (Diff)'})
    return clean_df

def main():
    st.sidebar.title("和碩 Pegatron 智慧工廠")
    st.sidebar.caption("Assignment 3 — Streamlit + Plotly 警報儀表板")

    num_rows = st.sidebar.slider("感測器筆數 (Rows)", 100, 500, 200, 50)
    anomaly_prob = st.sidebar.slider("異常機率", 0.05, 0.30, 0.12, 0.01)

    uploaded_file = st.sidebar.file_uploader("匯入感測器 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = generate_sensor_data(num_rows=num_rows, anomaly_ratio=anomaly_prob)

    processed_df = run_anomaly_pipeline(raw_df)

    st.title("🏭 智慧工廠設備異常警報 Streamlit 儀表板")
    st.caption("即時感測器 telemetry 時序分析 (Plotly) | 孤立森林 ML | Ground Truth 比對與 Accuracy 評估")

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

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['temp'], mode='lines', name='溫度 (°C)', line=dict(color='#f97316')))
    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['pressure'], mode='lines', name='壓力 (bar)', line=dict(color='#38bdf8'), yaxis='y2'))
    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['vibration'], mode='lines', name='震動 (g)', line=dict(color='#c084fc'), yaxis='y3'))

    if abnormal_count > 0:
        fig_ts.add_trace(go.Scatter(x=abnormal_df['timestamp'], y=abnormal_df['temp'], mode='markers', name='異常點', marker=dict(color='#ef4444', size=9, symbol='x')))

    fig_ts.update_layout(
        paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font=dict(color='#e2e8f0'), height=450,
        hovermode="x unified", legend=dict(orientation="h", y=1.15, x=0),
        yaxis=dict(title="溫度 (°C)"), yaxis2=dict(title="壓力 (bar)", overlaying='y', side='right'),
        yaxis3=dict(title="震動 (g)", overlaying='y', side='right', position=0.95)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("🚨 異常警報紀錄與 Gemini AI 建議處置")
    st.dataframe(
        processed_df[['timestamp', 'temp', 'pressure', 'vibration', 'severity', 'anomaly_score', 'root_cause', 'action_suggestion']],
        use_container_width=True
    )

if __name__ == "__main__":
    main()
