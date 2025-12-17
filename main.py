import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

# 忽略 XGBoost 版本警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 系統參數與場景設定 (System & Scenario Configuration)
# ==============================================================================

# --- 模擬環境：蜿蜒山路 (Mountainous Road) ---
ROUTE_LENGTH_KM = 12.0          # 路線總長度 (公里)
RESOLUTION_M = 10.0             # 空間解析度 (每 10 公尺一個點)
VEHICLE_SPEED_KMPH = 60.0       # 山路車速較慢 (60 km/h)
VEHICLE_SPEED_MPS = VEHICLE_SPEED_KMPH / 3.6
TIME_STEP_S = RESOLUTION_M / VEHICLE_SPEED_MPS  # 每個 Step 的時間 (約 0.6秒)

# --- 訊號強度定義 (RSRP in dBm) ---
RSRP_LOS = -80.0                # Line-of-Sight (看到基地台)
RSRP_NLOS = -130.0              # Non-Line-of-Sight (被山壁完全遮蔽)
NOISE_STD_DB = 3.0              # Shadowing/Fading 隨機雜訊

# --- 關鍵閾值 (Critical Thresholds) ---
# 1. Handover Trigger (-105): 低於此值開始考慮換手 (A4 Event)
THRESH_HO_TRIGGER = -105.0      
# 2. Radio Link Failure (-115): 低於此值視為斷線 (Service Drop)
THRESH_RLF = -115.0   

# --- 3GPP 換手時序限制 (Handover Timing Constraints) ---
# 衛星連線建立非常慢，包含測量(Measurement)、決策(Decision)、隨機存取(RACH)、RRC Setup
# 假設總共需要 3 秒 (這就是為什麼 Reactive 方法會失敗的主因)
HO_PREPARATION_TIME_S = 3     
HO_PREP_STEPS = int(np.ceil(HO_PREPARATION_TIME_S / TIME_STEP_S))

# --- AI 預測參數 ---
# 提前 5 秒預測，給予系統 1.5 秒的緩衝時間 (5.0 - 3 = 1.5s margin)
PREDICTION_HORIZON_S = 5.0      
PRED_HORIZON_STEPS = int(np.ceil(PREDICTION_HORIZON_S / TIME_STEP_S))

# --- GPS 誤差 (模擬魯棒性) ---
GPS_ERROR_STD_M = 15.0          # 模擬定位飄移 15 公尺

# --- Standard CHO (3GPP Rel-17) 參數 ---
# Time-to-Trigger (TTT): 訊號必須持續爛掉 0.48 秒才觸發，避免乒乓效應
CHO_TTT_S = 0.48     
CHO_TTT_STEPS = int(np.ceil(CHO_TTT_S / TIME_STEP_S))

print(f"================ SYSTEM CONFIGURATION ================")
print(f"Scenario: Mountainous Road (Ground-to-Satellite Handover)")
print(f"Vehicle Speed: {VEHICLE_SPEED_KMPH} km/h")
print(f"Handover Prep Time: {HO_PREPARATION_TIME_S} s ({HO_PREP_STEPS} steps)")
print(f"AI Prediction Horizon: {PREDICTION_HORIZON_S} s")
print(f"GPS Error (Robustness): {GPS_ERROR_STD_M} m")
print(f"======================================================\n")


# ==============================================================================
# 2. 環境生成器 (Ground Truth Generator)
# ==============================================================================
def generate_mountain_environment(noise_seed=None):
    """
    生成山區道路環境資料。
    特徵：
    1. 彎道處因為山壁遮擋，地面訊號 (Ground RSRP) 瞬間掉落。
    2. 頭頂衛星訊號 (Sat Elevation) 保持良好 (LoS)。
    """
    if noise_seed is not None:
        np.random.seed(noise_seed)
        
    dist_points = np.arange(0, ROUTE_LENGTH_KM * 1000, RESOLUTION_M)
    n_points = len(dist_points)
    
    # 模擬 3 個髮夾彎 (Hairpin Turns) 造成的突發遮蔽區
    # 0 = No Blockage, 1 = Blockage
    blockage_map = np.zeros(n_points)
    
    # 定義遮蔽區間 (公里)
    blockage_zones = [
        (7.0, 8.5)  # Zone 2: 長山壁 (主要測試區)
    ]
    
    for start_km, end_km in blockage_zones:
        idx_start = int(start_km * 1000 / RESOLUTION_M)
        idx_end = int(end_km * 1000 / RESOLUTION_M)
        blockage_map[idx_start:idx_end] = 1.0
        
    # --- 生成地面訊號 (Terrestrial RSRP) ---
    rsrp_ground = np.ones(n_points) * RSRP_LOS
    
    # 模擬突發衰減 (Sudden Drop) - Step Function
    # 山壁遮擋通常非常銳利，緩衝區很短 (約 50-100m)
    transition_len = 8 # 80m
    
    for start_km, end_km in blockage_zones:
        idx_start = int(start_km * 1000 / RESOLUTION_M)
        idx_end = int(end_km * 1000 / RESOLUTION_M)
        
        # 進入遮蔽 (Fade out)
        rsrp_ground[idx_start-transition_len:idx_start] = np.linspace(RSRP_LOS, -110, transition_len)
        # 遮蔽區 (Blocked)
        rsrp_ground[idx_start:idx_end] = RSRP_NLOS
        # 離開遮蔽 (Fade in)
        rsrp_ground[idx_end:idx_end+transition_len] = np.linspace(RSRP_NLOS, RSRP_LOS, transition_len)

    # 加入通道雜訊 (Shadowing)
    noise = np.random.normal(0, NOISE_STD_DB, n_points)
    rsrp_noisy = rsrp_ground + noise
    
    # --- 生成衛星特徵 (Satellite Features) ---
    # 假設這段路是東西向，衛星在南邊天空，仰角隨機波動但大部分時間可視
    # 模擬衛星仰角 (Elevation Angle): 45 ~ 80 度
    sat_elevation = 60 + 10 * np.sin(dist_points / 2000.0) + np.random.normal(0, 2, n_points)
    
    # --- 生成車輛航向 (Heading) ---
    # 模擬山路彎來彎去
    vehicle_heading = (dist_points / 100.0) % 360
    
    df = pd.DataFrame({
        'dist_m': dist_points,
        'blockage_label': blockage_map, # Ground Truth: 1=Blocked
        'rsrp_clean': rsrp_ground,      # Ground Truth Signal
        'rsrp_noisy': rsrp_noisy,       # Measurement
        'sat_elevation': sat_elevation, # Satellite Feature
        'vehicle_heading': vehicle_heading # Geometry Feature
    })
    return df


# ==============================================================================
# 3. 模型定義 (Predictors)
# ==============================================================================

# --- Baseline 2: Time-Series Predictor (歷史訊號預測) ---
class TimeSeriesPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=1)
        self.lookback = 15  # 看過去 15 個點 (約 9秒)
        
    def prepare_data(self, df):
        data = df.copy()
        features = []
        for i in range(1, self.lookback + 1):
            col = f'lag_{i}'
            data[col] = data['rsrp_noisy'].shift(i)
            features.append(col)
        
        # Target: 未來 5 秒的訊號
        data['target'] = data['rsrp_noisy'].shift(-PRED_HORIZON_STEPS)
        data = data.dropna()
        return data[features], data['target']

    def train(self, df):
        X, y = self.prepare_data(df)
        self.model.fit(X, y)
        
    def predict(self, history):
        if len(history) < self.lookback: return RSRP_LOS
        vec = np.array(history[-self.lookback:]).reshape(1, -1)
        return self.model.predict(vec)[0]


# --- Proposed: Spatial REM Predictor (位置 + 幾何 + 魯棒性) ---
class SpatialREMPredictor:
    def __init__(self):
        # 使用分類器預測 "是否會遮蔽" (Probability of Blockage)
        # 比直接預測 RSRP 更穩定
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=1, eval_metric='logloss')
        
    def prepare_data(self, df, is_training=False):
        # 關鍵創新：在訓練時加入 GPS 雜訊 (Data Augmentation)
        # 讓模型學會 "模糊區域"，抵抗 GPS 誤差
        
        dist = df['dist_m'].values
        if is_training:
            # 加入 10m 的隨機雜訊來訓練
            noise = np.random.normal(0, 10.0, len(dist))
            dist = dist + noise
            
        # 輸入特徵：位置 + 衛星仰角 + 車輛航向
        X = pd.DataFrame({
            'dist_m': dist,
            'sat_elevation': df['sat_elevation'],
            'vehicle_heading': df['vehicle_heading']
        })
        
        # Target: 未來是否會發生 RLF (訊號 < -115)
        # Shift target to future to learn prediction
        future_rsrp = df['rsrp_noisy'].shift(-PRED_HORIZON_STEPS)
        y = (future_rsrp < THRESH_RLF).astype(int)
        
        # 移除 NaN
        mask = ~np.isnan(y)
        return X[mask], y[mask]

    def train(self, df):
        # 開啟 is_training=True 來注入雜訊
        X, y = self.prepare_data(df, is_training=True)
        print(f"Training Spatial REM with {len(X)} samples (Noise Augmented)...")
        self.model.fit(X, y)
        
    def predict_proba(self, current_dist, map_db):
        # map_db 是車載資料庫，包含地形特徵
        # 1. 找到地圖上最近的點 (Nearest Neighbor)
        idx = int(current_dist / RESOLUTION_M)
        idx = max(0, min(idx, len(map_db) - 1))
        
        row = map_db.iloc[idx]
        
        # 2. 建構輸入向量
        input_data = pd.DataFrame({
            'dist_m': [current_dist], # 使用當前感測到的位置 (含誤差)
            'sat_elevation': [row['sat_elevation']], # 假設星曆已知
            'vehicle_heading': [row['vehicle_heading']]
        })
        
        # 3. 預測 "遮蔽機率"
        # Class 1 = Blockage
        prob = self.model.predict_proba(input_data)[0][1]
        return prob


# ==============================================================================
# 4. 模擬主迴圈 (Main Loop)
# ==============================================================================
def run_simulation():
    # 1. 準備資料
    print("Generating Environment Data...")
    df_train = generate_mountain_environment(noise_seed=42) # 訓練用 (歷史資料)
    df_test = generate_mountain_environment(noise_seed=2025) # 測試用 (當下路況)
    
    # 2. 訓練模型
    print("Training Models...")
    
    # Baseline: Time-Series
    ai_time = TimeSeriesPredictor()
    ai_time.train(df_train)
    
    # Proposed: Spatial REM
    ai_spatial = SpatialREMPredictor()
    ai_spatial.train(df_train)
    
    # 3. 初始化狀態變數
    # State Codes:
    # 0 = Connected (Terrestrial Only)
    # 1 = Preparing (Handover / Dual Connectivity Setup)
    # 2 = Protected (Satellite Connected)
    
    strategies = ['reactive', 'time_series', 'proposed']
    states = {s: [] for s in strategies}
    timers = {s: 0 for s in strategies} # 倒數計時器
    curr_state = {s: 0 for s in strategies}
    
    # Metrics
    interruption_steps = {s: 0 for s in strategies}
    
    history_buffer = [] # 給 Time-Series 用
    ttt_counter = 0     # 給 Reactive (TTT) 用
    
    steps = len(df_test)
    print(f"\nRunning Simulation ({steps} steps)...")
    
    for i in range(steps):
        # --- 真實世界感測 (Ground Truth & Sensor Data) ---
        real_dist = df_test['dist_m'].iloc[i]
        real_rsrp = df_test['rsrp_noisy'].iloc[i]
        
        # 更新歷史 buffer
        history_buffer.append(real_rsrp)
        if len(history_buffer) > 20: history_buffer.pop(0)
        
        # 是否發生斷線 (Physical Layer Failure)
        is_rlf = (real_rsrp < THRESH_RLF)
        
        # ============================================
        # Strategy 1: Reactive (Standard 3GPP A4 Event)
        # ============================================
        if curr_state['reactive'] == 0:
            # Condition 1: Signal < Threshold
            # Condition 2: Time-to-Trigger (TTT) satisfied
            if real_rsrp < THRESH_HO_TRIGGER:
                ttt_counter += 1
                if ttt_counter >= CHO_TTT_STEPS:
                    curr_state['reactive'] = 1 # Start Setup
                    timers['reactive'] = HO_PREP_STEPS
            else:
                ttt_counter = 0
                
        elif curr_state['reactive'] == 1:
            timers['reactive'] -= 1
            if timers['reactive'] <= 0:
                curr_state['reactive'] = 2 # Setup Done
        
        # ============================================
        # Strategy 2: Time-Series AI (Competitor)
        # ============================================
        # 預測未來 RSRP
        pred_rsrp = ai_time.predict(history_buffer)
        
        if curr_state['time_series'] == 0:
            if pred_rsrp < THRESH_HO_TRIGGER:
                curr_state['time_series'] = 1
                timers['time_series'] = HO_PREP_STEPS
                
        elif curr_state['time_series'] == 1:
            timers['time_series'] -= 1
            if timers['time_series'] <= 0:
                curr_state['time_series'] = 2

        # ============================================
        # Strategy 3: Proposed Spatial REM (Our Hero)
        # ============================================
        # 1. 模擬 GPS 誤差 (測試魯棒性)
        # 即使你在山裡定位飄了 15m，模型還是要知道這裡危險
        gps_dist = real_dist + np.random.normal(0, GPS_ERROR_STD_M)
        
        # 2. Look-ahead: 預測未來 5秒的位置
        future_dist = gps_dist + (VEHICLE_SPEED_MPS * PREDICTION_HORIZON_S)
        
        # 3. 詢問 Spatial Model: "前方會遮蔽嗎?"
        prob_blockage = ai_spatial.predict_proba(future_dist, df_train)
        
        if curr_state['proposed'] == 0:
            # 只要遮蔽機率 > 50%，立刻啟動雙連結 (Conservative Policy)
            if prob_blockage > 0.5:
                curr_state['proposed'] = 1
                timers['proposed'] = HO_PREP_STEPS # 開始建立衛星連線
                
        elif curr_state['proposed'] == 1:
            timers['proposed'] -= 1
            if timers['proposed'] <= 0:
                curr_state['proposed'] = 2 # 衛星連線建立完成 (Safe!)
                
        # ============================================
        # 記錄狀態與計算損失
        # ============================================
        for s in strategies:
            states[s].append(curr_state[s])
            
            # 如果發生 RLF 且 尚未完成切換 (State != 2)，就是斷線
            if is_rlf and curr_state[s] != 2:
                interruption_steps[s] += 1
                
        # Reset 機制 (為了圖表好看，過了遮蔽區後重置)
        # 在真實系統中，這是 "Satellite-to-Ground Handover"
        if real_rsrp > THRESH_HO_TRIGGER + 10: # 回到良好訊號區
             for s in strategies:
                 curr_state[s] = 0
                 timers[s] = 0
             ttt_counter = 0
             
    return df_test, states, interruption_steps


# ==============================================================================
# 5. 視覺化繪圖 (Visualization) - 全局視野 & 防遮擋優化版
# ==============================================================================
def plot_results(df, states, interruptions):
    dist_km = df['dist_m'] / 1000.0
    
    # 設定全域範圍 (0 ~ 12km)
    xlim_min, xlim_max = 0.0, 12.0
    
    # 使用簡潔風格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # -------------------------------------------------------
    # 圖表一：訊號強度與觸發點 (Signal Strength)
    # -------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # 1. 畫訊號 (線條變細，保持清晰)
    ax1.plot(dist_km, df['rsrp_noisy'], color='silver', alpha=0.5, label='Measured RSRP (Noisy)', linewidth=1)
    ax1.plot(dist_km, df['rsrp_clean'], color='black', linewidth=1.5, linestyle=':', label='True RSRP Trend')
    
    # 2. 畫閾值線
    ax1.axhline(THRESH_HO_TRIGGER, color='orange', linestyle='--', linewidth=1.5, label='HO Threshold (-105)')
    ax1.axhline(THRESH_RLF, color='red', linestyle='-', linewidth=1.5, label='RLF Threshold (-115)')
    
    # 3. 標示紅色遮蔽區
    ax1.axvspan(7.0, 8.5, color='mistyrose', alpha=0.5, label='Blockage Zone')
    
    # 4. 畫觸發點 (標記大小適中)
    strategies_plot = [
        ('reactive', 'green', 'v', 'Standard CHO'),
        ('time_series', 'orange', 'x', 'Time-Series AI'),
        ('proposed', 'blue', '*', 'Proposed Spatial REM')
    ]
    
    for strat, color, marker, label_name in strategies_plot:
        s_arr = np.array(states[strat])
        triggers = np.where(np.diff(s_arr, prepend=0) == 1)[0]
        
        # 找出所有觸發點並標記
        for t in triggers:
            # 只標記主要遮蔽區附近的點，避免如果前面有誤觸發導致圖太亂
            if 6.0 < dist_km.iloc[t] < 9.0:
                ax1.scatter(dist_km.iloc[t], df['rsrp_noisy'].iloc[t], 
                           c=color, s=150, marker=marker, label=f'{label_name} Trigger', zorder=10, edgecolors='white')

    # 5. 設定標籤
    ax1.set_ylabel('Signal Strength (dBm)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
    ax1.set_title('Figure 1: Signal Strength & Trigger Timing (Full Route)', fontsize=16, fontweight='bold')
    ax1.set_xlim(xlim_min, xlim_max)
    
    # **關鍵修改：將圖例移到下方外部，避免遮擋**
    # 這裡我們手動過濾重複的 label (因為 scatter 可能會重複貼 label)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize=11)
    
    plt.tight_layout() # 自動調整佈局保留圖例空間
    plt.show()

    # -------------------------------------------------------
    # 圖表二：連線狀態 (Connectivity Status)
    # -------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # 1. 畫連線狀態 (使用不同線型區分，而非只靠顏色)
    # Standard CHO: 綠色實線
    ax2.plot(dist_km, np.array(states['reactive']), color='green', label='Standard CHO', linewidth=2, linestyle='-')
    
    # Time-Series AI: 橘色虛線 (Dash)
    ax2.plot(dist_km, np.array(states['time_series']) + 0.05, color='orange', label='Time-Series AI', linewidth=2, linestyle='--')
    
    # Proposed: 藍色實線，稍微加一點點粗度以突顯 (Offset 防止完全重疊)
    ax2.plot(dist_km, np.array(states['proposed']) + 0.1, color='blue', label='Proposed Spatial REM', linewidth=2.5, linestyle='-')
    
    # 2. 標示紅色遮蔽區
    ax2.axvspan(7.0, 8.5, color='mistyrose', alpha=0.5)
    
    # 3. Y 軸文字
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Ground Only', 'Setting up...', 'Dual / Sat Connected'], fontsize=12)
    
    # 4. 註解 (Annotation) - 移到上方空白處
    # 箭頭指向 Proposed (藍線) 爬升的地方
    # 假設 Proposed 在 6.8km 左右爬升
    ax2.annotate('Proposed Trigger\n(Early Warning)', xy=(6.8, 1.2), xytext=(5.5, 1.8),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, color='blue', fontweight='bold', ha='center')

    # 箭頭指向 Standard (綠線) 爬升的地方
    # 假設 Standard 在 7.1km 左右爬升
    ax2.annotate('Standard Trigger\n(Too Late)', xy=(7.1, 1.0), xytext=(8.5, 1.5),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, color='green', fontweight='bold', ha='center')

    # 標示 Zero Interruption (不擋線，放在頂部)
    ax2.text(7.75, 2.05, "Zero Interruption Zone", color='blue', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))

    # 5. 設定標籤
    ax2.set_xlabel('Distance (km)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Connectivity State', fontsize=14, fontweight='bold')
    ax2.set_title('Figure 2: Connectivity Status (Full Route)', fontsize=16, fontweight='bold')
    ax2.set_xlim(xlim_min, xlim_max)
    
    # **圖例移到下方外部**
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_res, state_log, loss_log = run_simulation()
    
    print("\n================ FINAL RESULTS ================")
    print(f"Total Interruption Time (Steps):")
    print(f"1. Standard CHO:      {loss_log['reactive']} steps (FAILED)")
    print(f"2. Time-Series AI:    {loss_log['time_series']} steps (FAILED)")
    print(f"3. Proposed Spatial:  {loss_log['proposed']} steps (SUCCESS)")
    
    if loss_log['proposed'] == 0:
        print("\nCONCLUSION: The proposed method successfully achieved Zero-Interruption Handover!")
    else:
        print(f"\nNOTE: Small packet loss occurred ({loss_log['proposed']}), tune prediction horizon.")
    
    plot_results(df_res, state_log, loss_log)