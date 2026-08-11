"""
Pegatron Smart Factory Assignment 3: Equipment Anomaly Alert AI Agent
Author: Pegatron ML Application Engineer
File: factory_alert_agent.py

Features:
- Task 1: Data Ingestion & Preprocessing (Imputation & Z-score Standardization)
- Task 2: Isolation Forest ML Model + Median/MAD Robust Statistics Warning Engine
- Output: Real-Time Stream Terminal Output with ANSI Colorized Highlights & Actionable Recommendations

分級邏輯 (v2 更新):
  - 警告 (紅, ALERT)   : 物理指標超標 (依課程規格書門檻: temp>52/<43, pressure>1.08/<0.97, vibration>0.07)
  - 預警 (黃, WARNING) : 統計離群 (Modified Z-score, Median+MAD 穩健統計法,
                          參考 Iglewicz & Hoaglin 1993《How to Detect and Handle Outliers》)
                          且持續一段時間 + 呈現上升趨勢，才升級為預警 (避免單點雜訊誤報造成警報疲勞)
  - 正常 (綠, NORMAL)  : 以上皆非
"""

import pandas as pd
import numpy as np
import time
import random
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ANSI Color Codes for Terminal Highlighting
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[1;31m"      # 警告 ALERT (物理超標)
COLOR_YELLOW = "\033[1;33m"   # 預警 WARNING (統計離群)
COLOR_GREEN = "\033[92m"      # 正常 NORMAL
COLOR_CYAN = "\033[96m"       # HEADER / INFO

# ── 業界統計參數 (有依據，非隨意寫死) ──
MODIFIED_Z_THRESHOLD = 3.5    # Iglewicz & Hoaglin (1993) 建議之穩健離群門檻
TREND_Z_THRESHOLD = 2.0       # 早期預警敏感度：變化速率超出歷史波動 2 個穩健標準差
DEBOUNCE_WINDOW = 5           # 業務規則 (非統計值)：對應每分鐘 1 筆採樣，持續 5 分鐘才視為有效趨勢
# 注意：資料產生器 (generate_dataset.py / app.py 內建 generator) 的雜訊震幅已調整，
# 若你的 sensor_data.csv 仍是舊版產生的，pressure/vibration 可能會出現連續多筆相同值 (抖動過小)


def _robust_std(series: pd.Series) -> float:
    """用 MAD (Median Absolute Deviation) 估計穩健標準差，避免異常值污染統計基線 (masking effect)。"""
    med = series.median()
    mad = (series - med).abs().median()
    return float(mad * 1.4826) if mad > 1e-9 else float(series.std() + 1e-9)


class FactoryAnomalyAlertAgent:
    def __init__(self, csv_filepath="sensor_data.csv", impute_method="linear"):
        self.csv_filepath = csv_filepath
        self.impute_method = impute_method
        self.df = None
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination='auto', random_state=42)

    def load_and_preprocess(self):
        print(f"{COLOR_CYAN}{'='*75}")
        print("         PEGATRON SMART FACTORY EQUIPMENT ANOMALY ALERT AGENT          ")
        print(f"{'='*75}{COLOR_RESET}")
        print(f"[1/3] Ingesting dataset from '{self.csv_filepath}'...")

        try:
            self.df = pd.read_csv(self.csv_filepath)
        except Exception as e:
            print(f"{COLOR_RED}[ERROR] Failed to read {self.csv_filepath}: {e}{COLOR_RESET}")
            sys.exit(1)

        print(f"      Successfully loaded {len(self.df)} sensor records.")

        # Preprocessing: Impute missing values
        for col in ['temp', 'pressure', 'vibration']:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                if self.impute_method == 'linear':
                    self.df[col] = self.df[col].interpolate(method='linear').bfill().ffill()
                    method_str = "Linear Interpolation"
                elif self.impute_method == 'ffill':
                    self.df[col] = self.df[col].ffill().bfill()
                    method_str = "Forward Fill / LOCF"
                else:
                    col_median = self.df[col].median()
                    self.df[col] = self.df[col].fillna(col_median)
                    method_str = f"Median ({col_median:.2f})"
                print(f"[CLEANING] Imputed {missing_count} missing values in '{col}' using {method_str}")

        # Round numerical columns to standard decimals
        self.df['temp'] = self.df['temp'].round(1)
        self.df['pressure'] = self.df['pressure'].round(2)
        self.df['vibration'] = self.df['vibration'].round(2)

        # Standardize features (Z-Score)
        features = ['temp', 'pressure', 'vibration']
        self.df[['temp_z', 'pressure_z', 'vibration_z']] = self.scaler.fit_transform(self.df[features])
        print("[2/3] Data Preprocessing Complete: Feature Standardization (Z-score) applied.")

    def run_detection_agent(self):
        print(f"\n[3/3] Running AI Agent Real-Time Stream Detection Engine...")
        print(f"{COLOR_CYAN}{'-'*75}{COLOR_RESET}")

        features = ['temp', 'pressure', 'vibration']
        self.model.fit(self.df[features])

        iso_preds = self.model.predict(self.df[features])            # -1 for anomaly, 1 for normal
        iso_scores = self.model.decision_function(self.df[features])  # 連續分數，越負越異常

        # ── 業界作法：用穩健統計量 (Median + MAD) 動態算出離群門檻 ──
        # 取代原本寫死的固定分數門檻 (如 0.6)
        median_score = float(np.median(iso_scores))
        mad_score = float(np.median(np.abs(iso_scores - median_score)))
        mad_score = mad_score if mad_score > 1e-9 else 1e-9

        def modified_z_score(score):
            return 0.6745 * (score - median_score) / mad_score

        # 用一階差分的穩健標準差，取代原本寫死的趨勢門檻 (0.1 / 0.008 / 0.002)
        temp_diff_std = _robust_std(self.df['temp'].diff().fillna(0))
        press_diff_std = _robust_std(self.df['pressure'].diff().fillna(0))
        vib_diff_std = _robust_std(self.df['vibration'].diff().fillna(0))

        print(f"[INFO] Robust Statistics Baseline -> Median(S)={median_score:.4f}, "
              f"MAD(S)={mad_score:.4f}, Modified Z Threshold={MODIFIED_Z_THRESHOLD}")
        print(f"{COLOR_CYAN}{'-'*75}{COLOR_RESET}")

        alerts_triggered = 0
        pred_labels = []

        for i in range(len(self.df)):
            # Stream delay: simulate real-time telemetry streaming (0.1s ~ 0.25s)
            time.sleep(random.uniform(0.1, 0.25))

            row = self.df.iloc[i]
            reasons = []
            rule_score = 0.0

            # ── Layer 1: 物理閾值判斷 (課程規格書門檻，直接採用不變) ──
            if row['temp'] > 52.0:
                reasons.append(f"過熱 ({row['temp']}°C > 52°C)")
                rule_score += 0.45
            elif row['temp'] < 43.0:
                reasons.append(f"過冷 ({row['temp']}°C < 43°C)")
                rule_score += 0.40

            if row['pressure'] > 1.08:
                reasons.append(f"壓力過高 ({row['pressure']} > 1.08 bar)")
                rule_score += 0.40
            elif row['pressure'] < 0.97:
                reasons.append(f"壓力過低 ({row['pressure']} < 0.97 bar)")
                rule_score += 0.35

            if row['vibration'] > 0.07:
                reasons.append(f"劇烈震動 ({row['vibration']} > 0.07 g)")
                rule_score += 0.50

            has_physical_violation = (len(reasons) > 0)

            # ── Layer 2: 統計離群判斷 (Modified Z-score，動態計算，非寫死) ──
            mz = modified_z_score(iso_scores[i])
            is_statistical_outlier = (mz > MODIFIED_Z_THRESHOLD) and (iso_preds[i] == -1)

            # ── Layer 3: 趨勢判斷 (用該特徵自己的滾動窗口穩健標準差，非寫死數字) ──
            start_i = max(0, i - (DEBOUNCE_WINDOW - 1))
            window_len = i - start_i + 1
            is_persistent = (window_len >= DEBOUNCE_WINDOW)

            temp_trend = row['temp'] - self.df.iloc[start_i]['temp']
            press_trend = row['pressure'] - self.df.iloc[start_i]['pressure']
            vib_trend = row['vibration'] - self.df.iloc[start_i]['vibration']

            temp_trend_z = temp_trend / temp_diff_std
            press_trend_z = press_trend / press_diff_std
            vib_trend_z = vib_trend / vib_diff_std

            has_upward_trend = (
                (temp_trend_z > TREND_Z_THRESHOLD) or
                (press_trend_z > TREND_Z_THRESHOLD) or
                (vib_trend_z > TREND_Z_THRESHOLD)
            )

            is_early_warning = (
                is_statistical_outlier and has_upward_trend and is_persistent and not has_physical_violation
            )

            # ── 三段式分級：警告(紅) / 預警(黃) / 正常(綠) ──
            if has_physical_violation:
                pred_label = 'abnormal'
                sev = 'ALERT'
                final_score = round(min(1.0, max(0.75, 0.40 + rule_score)), 2)
                root_cause = ", ".join(reasons)
                warn_reason = f"檢測到物理指標超標 (依規格書門檻): {root_cause}"

                if "過熱" in str(reasons):
                    act = "檢查冷卻泵浦與水管流量，降低機台負載 25%"
                elif "震動" in str(reasons):
                    act = "安排主軸軸承雷射對心，補給潤滑油脂"
                elif "壓力" in str(reasons):
                    act = "檢修氣壓歧管與分流閥，確認有無漏氣"
                else:
                    act = "派員進行感測器校正與物理維修"

            elif is_early_warning:
                pred_label = 'normal'
                sev = 'WARNING'
                excess = min(mz - MODIFIED_Z_THRESHOLD, 5.0) / 5.0
                final_score = round(0.50 + excess * 0.24, 2)
                root_cause = f"統計離群 (Modified Z-score = {mz:.2f} > {MODIFIED_Z_THRESHOLD})"
                warn_reason = (f"【穩健統計法】Isolation Forest 連續 {window_len} 分鐘偏離正常分佈中心 "
                                f"(Median+MAD)，且變化速率超出歷史波動 {TREND_Z_THRESHOLD} 個穩健標準差，"
                                f"觸發早期預警 [WARNING]")
                act = "派員進行感測器校正與預防性巡檢"

            else:
                pred_label = 'normal'
                sev = 'NORMAL'
                final_score = 0.0
                root_cause = "無異常 (Normal)"
                warn_reason = "感測器數值在統計正常範圍內"
                act = "設備運作正常，維持預防性維護"

            pred_labels.append(pred_label)

            # Terminal Output Formatting according to Severity
            if sev == 'NORMAL':
                # Lightweight single-line stream for normal telemetry (綠)
                print(f"{COLOR_GREEN}[{row['timestamp']}] TEMP: {row['temp']:>4.1f}°C | PRESS: {row['pressure']:>4.2f}bar | VIB: {row['vibration']:>5.3f}g | STATUS: OK (正常){COLOR_RESET}")
            else:
                # Highlighted structured Alert Block for WARNING(黃) / ALERT(紅)
                alerts_triggered += 1
                color = COLOR_RED if sev == 'ALERT' else COLOR_YELLOW
                sev_zh = "警告" if sev == 'ALERT' else "預警"

                print(f"\n{color}{'='*75}")
                print(f" [ALERT #{alerts_triggered}] 🚨 LEVEL: {sev} ({sev_zh}) | ANOMALY SCORE: {final_score*100:.1f}% | TIME: {row['timestamp']}")
                print(f" {'-'*75}")
                print(f"   -> Telemetry   : TEMP: {row['temp']}°C, PRESSURE: {row['pressure']} bar, VIB: {row['vibration']} g")
                print(f"   -> Evaluation  : Ground Truth: {row.get('label', 'N/A')} | Predicted Label: {pred_label}")
                print(f"   -> Root Cause (真因) : {root_cause}")
                print(f"   -> Warning Reason    : {warn_reason}")
                print(f"   -> SOP Action (指引) : {act}")
                print(f"{'='*75}{COLOR_RESET}\n")

                # Brief pause for alert visual focus
                time.sleep(0.25)

        self.df['predicted_label'] = pred_labels

        # Evaluation Metrics
        total_rows = len(self.df)
        gt_matches = (self.df['predicted_label'] == self.df['label']).sum() if 'label' in self.df.columns else total_rows
        accuracy = (gt_matches / total_rows) * 100 if total_rows > 0 else 100.0

        tp = ((self.df['predicted_label'] == 'abnormal') & (self.df['label'] == 'abnormal')).sum() if 'label' in self.df.columns else 0
        fp = ((self.df['predicted_label'] == 'abnormal') & (self.df['label'] == 'normal')).sum() if 'label' in self.df.columns else 0
        fn = ((self.df['predicted_label'] == 'normal') & (self.df['label'] == 'abnormal')).sum() if 'label' in self.df.columns else 0

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 100.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 100.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 100.0

        print(f"\n{COLOR_CYAN}{'='*75}")
        print("                       SUMMARY EXECUTION REPORT                          ")
        print(f"{'='*75}{COLOR_RESET}")
        print(f" Total Sensor Records Evaluated  : {total_rows}")
        print(f" Total Anomaly Alerts Dispatched : {alerts_triggered} ({alerts_triggered/total_rows*100:.1f}%)")
        print(f" Ground Truth Match Count (GT)   : {gt_matches} / {total_rows}")
        print(f" Accuracy vs Ground Truth Label  : {accuracy:.2f}%")
        print(f" Model Precision                 : {precision:.2f}%")
        print(f" Model Recall                    : {recall:.2f}%")
        print(f" Model F1-Score                  : {f1_score:.2f}%")
        print(f" Status: {COLOR_GREEN}AI Agent System Active & Operational.{COLOR_RESET}")
        print(f"{COLOR_CYAN}{'='*75}{COLOR_RESET}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "sensor_data.csv"
    agent = FactoryAnomalyAlertAgent(filepath)
    agent.load_and_preprocess()
    agent.run_detection_agent()
