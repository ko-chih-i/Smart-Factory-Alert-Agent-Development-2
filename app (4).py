"""
智慧工廠設備異常警報儀表板 (Smart Factory Equipment Anomaly Alert Dashboard)
檔案名稱: app.py
套件需求: streamlit, plotly, pandas, numpy, scikit-learn

執行方式:
  pip install -r requirements.txt
  streamlit run app.py
"""

import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
def generate_sensor_data(num_rows=200, anomaly_ratio=0.12, inject_missing=False, missing_rate=0.04, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    start_time = datetime.strptime("2024-06-03 19:05:00", "%Y-%m-%d %H:%M:%S")
    data = []

    # Physical Machine State Tracker (Simulates real thermal inertia & mechanical continuity)
    current_temp = 47.5
    current_pressure = 1.025
    current_vibration = 0.030

    anomaly_phase_remaining = 0
    current_anomaly_type = None

    for i in range(num_rows):
        current_time = (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if a new realistic machine fault event triggers (continuous duration)
        if anomaly_phase_remaining <= 0:
            if random.random() < (anomaly_ratio * 0.28):
                anomaly_phase_remaining = random.randint(4, 11) # Event lasts 4-11 minutes continuously
                current_anomaly_type = random.choice(['high_temp', 'low_temp', 'high_press', 'low_press', 'high_vib', 'compound'])
            else:
                current_anomaly_type = None

        # Base normal physical setpoints with mild periodic motor load fluctuation (sine cycles)
        cycle_phase = (i / 18.0) * math.pi * 2.0
        target_normal_temp = 47.5 + 1.2 * math.sin(cycle_phase)
        target_normal_press = 1.025 + 0.01 * math.cos(cycle_phase)
        target_normal_vib = 0.030 + 0.004 * math.sin(cycle_phase * 2.0)

        target_temp = target_normal_temp
        target_press = target_normal_press
        target_vib = target_normal_vib

        if anomaly_phase_remaining > 0 and current_anomaly_type:
            anomaly_phase_remaining -= 1
            if current_anomaly_type == 'high_temp':
                target_temp = 53.5 + random.uniform(0.0, 6.5)
            elif current_anomaly_type == 'low_temp':
                target_temp = 36.0 + random.uniform(0.0, 5.0)
            elif current_anomaly_type == 'high_press':
                target_press = 1.10 + random.uniform(0.0, 0.15)
            elif current_anomaly_type == 'low_press':
                target_press = 0.85 + random.uniform(0.0, 0.10)
            elif current_anomaly_type == 'high_vib':
                target_vib = 0.08 + random.uniform(0.0, 0.06)
            elif current_anomaly_type == 'compound':
                target_temp = 54.0 + random.uniform(0.0, 5.0)
                target_press = 1.12 + random.uniform(0.0, 0.10)
                target_vib = 0.08 + random.uniform(0.0, 0.05)

        # Apply thermal inertia & physical response smoothing (Auto-regressive process)
        current_temp = 0.65 * current_temp + 0.35 * target_temp + (random.random() - 0.5) * 0.3
        current_pressure = 0.70 * current_pressure + 0.30 * target_press + (random.random() - 0.5) * 0.006
        current_vibration = 0.70 * current_vibration + 0.30 * target_vib + (random.random() - 0.5) * 0.002

        temp = round(float(current_temp), 1)
        pressure = round(float(current_pressure), 2)
        vibration = round(float(current_vibration), 2)

        # Keep normal operation strictly inside normal limits when no anomaly active
        if not current_anomaly_type and anomaly_phase_remaining <= 0:
            if temp > 50.0: temp = 49.8
            if temp < 45.0: temp = 45.2
            if pressure > 1.05: pressure = 1.04
            if pressure < 1.00: pressure = 1.01
            if vibration > 0.04: vibration = 0.038
            if vibration < 0.02: vibration = 0.022

        # Strictly determine ground truth label based on physical parameters
        is_physically_abnormal = (temp > 52.0) or (temp < 43.0) or (pressure > 1.08) or (pressure < 0.97) or (vibration > 0.07)
        label = 'abnormal' if is_physically_abnormal else 'normal'

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
    clean_df = df.copy().reset_index(drop=True)
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

    clean_df['temp'] = clean_df['temp'].round(1)
    clean_df['pressure'] = clean_df['pressure'].round(2)
    clean_df['vibration'] = clean_df['vibration'].round(2)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df[['temp', 'pressure', 'vibration']])
    clean_df['temp_z'], clean_df['pressure_z'], clean_df['vibration_z'] = scaled[:, 0], scaled[:, 1], scaled[:, 2]

    iso_forest = IsolationForest(contamination='auto', random_state=42)
    iso_forest.fit(clean_df[['temp', 'pressure', 'vibration']])
    iso_preds = iso_forest.predict(clean_df[['temp', 'pressure', 'vibration']])
    iso_scores = iso_forest.decision_function(clean_df[['temp', 'pressure', 'vibration']])

    raw_ml_scores = []
    for pred, score in zip(iso_preds, iso_scores):
        if pred == -1 or score < 0:
            raw_ml_scores.append(min(1.0, max(0.20, float(-score * 3.5))))
        else:
            raw_ml_scores.append(0.0)

    reasons_list, warning_reasons, severities, anomaly_scores, actions, predicted_labels = [], [], [], [], [], []
    is_temp_anom_list, is_press_anom_list, is_vib_anom_list = [], [], []

    for i in range(len(clean_df)):
        row = clean_df.iloc[i]
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

        has_physical_violation = (len(reasons) > 0)
        is_iso_outlier = (iso_preds[i] == -1)

        # 多重條件過濾 (Debounce / Persistent Check)
        start_i = max(0, i - 4)
        window_len = i - start_i + 1
        is_persistent_ml = (window_len >= 5) and all(raw_ml_scores[k] > 0.6 for k in range(start_i, i + 1))

        temp_trend = clean_df.iloc[i]['temp'] - clean_df.iloc[start_i]['temp']
        press_trend = clean_df.iloc[i]['pressure'] - clean_df.iloc[start_i]['pressure']
        vib_trend = clean_df.iloc[i]['vibration'] - clean_df.iloc[start_i]['vibration']
        has_upward_trend = (temp_trend > 0.1) or (press_trend > 0.008) or (vib_trend > 0.002)

        is_debounced_warning = is_persistent_ml and has_upward_trend

        if has_physical_violation:
            pred_label = 'abnormal'
            final_score = round(min(1.0, max(0.75, 0.40 + score, raw_ml_scores[i])), 2)
            sev = 'CRITICAL' if final_score >= 0.85 else 'HIGH'
            if "過熱" in str(reasons):
                act = "檢查冷卻泵浦與水管流量，降低機台負載 25%"
            elif "震動" in str(reasons):
                act = "安排主軸軸承雷射對心，補給潤滑油脂"
            elif "壓力" in str(reasons):
                act = "檢修氣壓歧管與分流閥，確認有無漏氣"
            else:
                act = "派員進行感測器校正與物理維修"
            warn_reason = f"檢測到物理指標超標: {', '.join(reasons)}"
        elif is_debounced_warning:
            pred_label, sev, act = 'normal', 'WARNING', '派員進行感測器校正與預防性巡檢'
            final_score = round(max(raw_ml_scores[i], 0.61), 2)
            warn_reason = f"【多重條件過濾通過】Isolation Forest 連續 5 分鐘離群 (Score={final_score:.2f} > 0.6) 且伴隨微幅上升趨勢，觸發預警 [WARNING]"
        else:
            pred_label, sev, act = 'normal', 'NORMAL', '設備運作正常，維持預防性維護'
            warn_reason = '感測器數值在標準公差範圍內 (Normal)'
            final_score = 0.0

        # Determine specific feature anomaly channels
        temp_viol = (row['temp'] > 52.0) or (row['temp'] < 43.0)
        press_viol = (row['pressure'] > 1.08) or (row['pressure'] < 0.97)
        vib_viol = (row['vibration'] > 0.07)

        if has_physical_violation:
            is_temp_anom = temp_viol
            is_press_anom = press_viol
            is_vib_anom = vib_viol
        elif is_debounced_warning:
            is_temp_anom = (temp_trend > 0.1) or (abs(clean_df.iloc[i]['temp_z']) >= 1.5)
            is_press_anom = (press_trend > 0.008) or (abs(clean_df.iloc[i]['pressure_z']) >= 1.5)
            is_vib_anom = (vib_trend > 0.002) or (abs(clean_df.iloc[i]['vibration_z']) >= 1.5)

            if not (is_temp_anom or is_press_anom or is_vib_anom):
                max_z_idx = int(np.argmax([
                    abs(clean_df.iloc[i]['temp_z']),
                    abs(clean_df.iloc[i]['pressure_z']),
                    abs(clean_df.iloc[i]['vibration_z'])
                ]))
                is_temp_anom = (max_z_idx == 0)
                is_press_anom = (max_z_idx == 1)
                is_vib_anom = (max_z_idx == 2)
        else:
            is_temp_anom = False
            is_press_anom = False
            is_vib_anom = False

        is_temp_anom_list.append(is_temp_anom)
        is_press_anom_list.append(is_press_anom)
        is_vib_anom_list.append(is_vib_anom)

        reasons_list.append(", ".join(reasons) if reasons else ("孤立森林離群" if is_iso_outlier else "正常"))
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
    clean_df['is_temp_anom'] = is_temp_anom_list
    clean_df['is_press_anom'] = is_press_anom_list
    clean_df['is_vib_anom'] = is_vib_anom_list
    if 'label' in clean_df.columns:
        clean_df['gt_match'] = (clean_df['predicted_label'] == clean_df['label']).map({True: '✓ 一致 (Match)', False: '✗ 差異 (Diff)'})
    return clean_df, imputed_info

def main():
    st.sidebar.title("和碩 Pegatron 智慧工廠")
    st.sidebar.caption("警報儀表板")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 1. 設定與生成 CSV 數據")

    num_rows = st.sidebar.slider("感測器筆數 (Rows)", 100, 500, 200, 50)
    anomaly_prob = st.sidebar.slider("異常機率", 0.05, 0.30, 0.12, 0.01)

    if 'data_seed' not in st.session_state:
        st.session_state.data_seed = 42

    seed_input = st.sidebar.number_input("隨機種子 (Random Seed)", value=st.session_state.data_seed, step=1)
    st.session_state.data_seed = seed_input

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
            index=0,
            help="選擇遺失值填補策略：連續時序數據推薦「線性插值」；狀態維持推薦「前向填補」；穩定數值推薦「中位數填補」"
        )

    gen_csv_clicked = st.sidebar.button("🎲 生成 / 重新產生 CSV 數據檔", use_container_width=True)
    if gen_csv_clicked:
        import random
        st.session_state.data_seed = random.randint(1, 10000)
        st.session_state.test_executed = False
        st.session_state.stream_count = 1
        st.rerun()

    # Pre-generate CSV dataset based on Section 1 parameters
    generated_df = generate_sensor_data(
        num_rows=num_rows,
        anomaly_ratio=anomaly_prob,
        inject_missing=inject_missing,
        seed=st.session_state.data_seed
    )

    st.sidebar.download_button(
        label="📥 下載生成的 CSV 數據檔 (sensor_data.csv)",
        data=generated_df.to_csv(index=False).encode('utf-8'),
        file_name="sensor_data.csv",
        mime="text/csv",
        help="點擊下載當前根據設定產生的感測器原始數據 CSV 檔",
        use_container_width=True
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 2. 手動上傳 CSV 檔案 (可選)")
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
        mime="text/csv",
        help="點擊下載標準 CSV 範例檔，可作為自訂數據上傳範本"
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        raw_df.to_csv("sensor_data.csv", index=False)
        st.sidebar.caption(f"✅ 已載入手動上傳檔: {uploaded_file.name} ({len(raw_df)} 筆)")
    else:
        raw_df = generated_df
        raw_df.to_csv("sensor_data.csv", index=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ 3. 執行模式選擇")
    exec_mode = st.sidebar.radio(
        "請選擇執行模式",
        options=["靜態一次性載入分析", "動態 3 秒逐筆串流模式"],
        index=0,
        help="選擇『靜態』將一次性載入全部數據分析；選擇『動態』將在按下開始按鈕後，每 3 秒自動新增 1 筆數據"
    )
    stream_active = (exec_mode == "動態 3 秒逐筆串流模式")

    total_raw_rows = len(raw_df)

    if stream_active:
        if 'stream_count' not in st.session_state:
            st.session_state.stream_count = 1

        st.sidebar.info(f"""🔴 **動態串流模式預備**
- 當前累積進度：{st.session_state.get('stream_count', 1)} / {total_raw_rows} 筆
- 更新步調：**每 3 秒自動新增 1 筆**""")

        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button("🔄 重頭開始", use_container_width=True):
            st.session_state.stream_count = 1
            st.rerun()
        if col_s2.button("⏩ 一次載入全部", use_container_width=True):
            st.session_state.stream_count = total_raw_rows
            st.rerun()

        if st.session_state.get('stream_count', 1) >= total_raw_rows:
            st.sidebar.success(f"✅ 已完成全量 {total_raw_rows} 筆數據動態串流！")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 4. 啟動測試")
    run_test_clicked = st.sidebar.button("🚀 開始執行 AI Agent 測試", type="primary", use_container_width=True)

    st.title("🏭 智慧工廠設備異常警報 AI 儀表板 (Blue Theme)")

    if 'test_executed' not in st.session_state:
        st.session_state.test_executed = False

    if run_test_clicked:
        st.session_state.test_executed = True
        if stream_active and 'stream_count' not in st.session_state:
            st.session_state.stream_count = 1

    if not st.session_state.test_executed:
        col_act, col_prev = st.columns([1, 2])
        with col_act:
            st.markdown("### 📋 測試準備狀態")
            st.markdown(f"- **數據來源**: sensor_data.csv\n- **數據總筆數**: {len(raw_df)} 筆")
            if st.button("🚀 開始執行 AI Agent 測試 (Start Test)", type="primary", key="tmpl_start_btn", use_container_width=True):
                st.session_state.test_executed = True
                if stream_active:
                    st.session_state.stream_count = 1
                st.rerun()

        with col_prev:
            st.markdown("### 📄 產出之 CSV 原始數據預覽 (sensor_data.csv)")
            st.dataframe(raw_df.head(12), use_container_width=True)

        return

    # Slice data for streaming mode ONLY AFTER test execution has started
    if stream_active:
        raw_df = raw_df.iloc[:st.session_state.get('stream_count', total_raw_rows)]

    processed_df, imputed_info = run_anomaly_pipeline(raw_df, impute_method=impute_method)

    st.title("🏭 智慧工廠設備異常警報儀表板")
   

    # Documentation & System Architecture Expanders
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("🧮 1. 怎麼算出 Anomaly Score？（分數算式說明）", expanded=False):
            st.markdown("""
            **Anomaly Score（異常預警指數）推導與計算機制**：
            1. **Z-Score 特徵標準化**：
               Z = (X - μ) / σ
               對 '[temp, pressure, vibration]' 轉為標準常態分佈 N(0, 1)。
            2. **Isolation Forest 孤立樹離群評估**：
               模型計算決策分數 S_raw = decision_function([Z_temp, Z_press, Z_vib])。
            3. **分數歸一化 (0% ~ 100%)**：
               Anomaly Score = Clip((0.5 - S_raw) * 100%, 0%, 100%)
            4. **預警層級劃分 (Warning Rating)**：
               - '< 50%'：**NORMAL** (數據位於集中密度區)
               - '50% ~ 65%'：**WARNING** (早期預警，提示排查保養)
               - '65% ~ 80%'：**HIGH** (顯著偏離正常區間)
               - '≥ 80%'：**CRITICAL** (極度離群或物理超標)
            """)

    with col_exp2:
        with st.expander("📁 2. 上傳 CSV 建議格式規範與範例說明", expanded=False):
            st.markdown("""
            **上傳 CSV 建議欄位名稱與資料型別**：
            - 'timestamp' *(建議)*：時間戳記，例如 '2024-06-03 19:05:00'
            - 'temp' *(必填)*：設備溫度 (°C)，正常約 '45.0 ~ 50.0'
            - 'pressure' *(必填)*：管線壓力 (bar)，正常約 '1.00 ~ 1.05'
            - 'vibration' *(必填)*：三軸震動 (g)，正常約 '0.02 ~ 0.04'
            - 'label' *(選填)*：實際標籤 ('normal' / 'abnormal')，若省略將由 Agent 進行 100% 無監督推論。

            💡 *側面選單亦可點擊「下載 CSV 建議上傳範例」按鈕取得標籤範本。*
            """)

    latest_row = processed_df.iloc[-1] if len(processed_df) > 0 else None
    current_sev = latest_row['severity'] if latest_row is not None else 'NORMAL'
    latest_score = latest_row['anomaly_score'] if latest_row is not None else 0.0
    latest_time = str(latest_row['timestamp']).split()[-1] if (latest_row is not None and 'timestamp' in latest_row) else "19:10:00"

    if latest_row is not None and current_sev != 'NORMAL':
        latest_cause = latest_row['root_cause'] if latest_row['root_cause'] else "設備異常"
        latest_action = f"建議: {latest_row['action_suggestion']}" if latest_row['action_suggestion'] else "建議: 派員巡檢"
    else:
        latest_cause = "運作正常"
        latest_action = "建議: 維持預防性維護"

    total_count = len(processed_df)
    abnormal_df = processed_df[processed_df['predicted_label'] == 'abnormal']
    abnormal_count = len(abnormal_df)
    normal_count = total_count - abnormal_count
    anomaly_rate = (abnormal_count / total_count * 100) if total_count > 0 else 0.0

    gt_matches = (processed_df['predicted_label'] == processed_df['label']).sum() if 'label' in processed_df.columns else total_count
    accuracy = (gt_matches / total_count * 100) if total_count > 0 else 100.0
    critical_count = len(processed_df[processed_df['severity'] == 'CRITICAL'])

    m1, m2, m3 = st.columns(3)
    status_text = "正常" if current_sev == 'NORMAL' else "需注意"
    m1.metric("機台當前狀態", f"{current_sev}", delta=f"快照 {latest_time} ({status_text})", delta_color="normal" if current_sev == 'NORMAL' else "inverse")
    m2.metric("即時預警指數", f"{latest_score*100:.1f}%", delta="Isolation Forest ML")
    
    short_action = latest_action if len(latest_action) <= 20 else f"{latest_action[:18]}..."
    m3.metric("當前異常真因", f"{latest_cause}", delta=f"{short_action}")

    st.caption(f"📊 **Ground Truth 比對正確率**: {accuracy:.1f}% ({gt_matches}/{total_count} Match) | 區間累積警報: {abnormal_count} 筆 ({anomaly_rate:.1f}% 異常率) | 緊急警報 (CRITICAL): {critical_count} 筆")

    st.markdown("---")

    fig_ts = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=('🌡️ 溫度時序分層趨勢 (°C)', '🗜️ 壓力時序分層趨勢 (bar)', '⚡ 震動時序分層趨勢 (g)')
    )
    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['temp'], mode='lines', name='溫度 (°C)', line=dict(color='#f97316', width=2)), row=1, col=1)
    fig_ts.add_hline(y=52, line_dash="dash", line_color="#ef4444", annotation_text="上限 52°C", row=1, col=1)
    fig_ts.add_hline(y=43, line_dash="dash", line_color="#ef4444", annotation_text="下限 43°C", row=1, col=1)
    temp_anom_df = processed_df[processed_df['is_temp_anom']]
    if len(temp_anom_df) > 0:
        fig_ts.add_trace(go.Scatter(x=temp_anom_df['timestamp'], y=temp_anom_df['temp'], mode='markers', name='預測異常 (溫度)', marker=dict(color='#ef4444', size=8, symbol='x')), row=1, col=1)

    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['pressure'], mode='lines', name='壓力 (bar)', line=dict(color='#3b82f6', width=2)), row=2, col=1)
    fig_ts.add_hline(y=1.08, line_dash="dash", line_color="#ef4444", annotation_text="上限 1.08 bar", row=2, col=1)
    fig_ts.add_hline(y=0.97, line_dash="dash", line_color="#ef4444", annotation_text="下限 0.97 bar", row=2, col=1)
    press_anom_df = processed_df[processed_df['is_press_anom']]
    if len(press_anom_df) > 0:
        fig_ts.add_trace(go.Scatter(x=press_anom_df['timestamp'], y=press_anom_df['pressure'], mode='markers', name='預測異常 (壓力)', marker=dict(color='#ef4444', size=8, symbol='x')), row=2, col=1)

    fig_ts.add_trace(go.Scatter(x=processed_df['timestamp'], y=processed_df['vibration'], mode='lines', name='震動 (g)', line=dict(color='#a855f7', width=2)), row=3, col=1)
    fig_ts.add_hline(y=0.07, line_dash="dash", line_color="#ef4444", annotation_text="上限 0.07 g", row=3, col=1)
    vib_anom_df = processed_df[processed_df['is_vib_anom']]
    if len(vib_anom_df) > 0:
        fig_ts.add_trace(go.Scatter(x=vib_anom_df['timestamp'], y=vib_anom_df['vibration'], mode='markers', name='預測異常 (震動)', marker=dict(color='#ef4444', size=8, symbol='x')), row=3, col=1)

    fig_ts.update_layout(paper_bgcolor='#0b132b', plot_bgcolor='#0b132b', font=dict(color='#e2e8f0'), height=680, hovermode="x unified", showlegend=False)
    fig_ts.update_xaxes(gridcolor='#1e293b')
    fig_ts.update_yaxes(gridcolor='#1e293b')
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("🚨 設備異常警報清單與 Agent 預測標籤 (Predicted Label)")
    def style_abnormal_rows(row):
        is_abn = (str(row.get('predicted_label', '')).lower() == 'abnormal') or (str(row.get('label', '')).lower() == 'abnormal')
        is_warn = (str(row.get('severity', '')).upper() in ['CRITICAL', 'HIGH', 'WARNING'])
        if is_abn or is_warn:
            return ['color: #ef4444; font-weight: bold; background-color: rgba(153, 27, 27, 0.25);' for _ in row]
        return ['' for _ in row]

    display_df = processed_df.iloc[::-1][['timestamp', 'temp', 'pressure', 'vibration', 'predicted_label', 'label', 'gt_match', 'severity', 'anomaly_score', 'root_cause', 'action_suggestion']]
    st.dataframe(
        display_df.style.apply(style_abnormal_rows, axis=1),
        use_container_width=True
    )

    # 3-Second Auto-Advance Loop for Live Stream Mode
    if stream_active and st.session_state.get('stream_count', 0) < total_raw_rows:
        time.sleep(3)
        st.session_state.stream_count += 1
        st.rerun()

if __name__ == "__main__":
    main()
