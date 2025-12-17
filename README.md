
---

### `README.md`


# 混合 NTN-地面網路中基於預測性雙連結啟動之切換機制

本儲存庫 (Repository) 包含期末專題 **「混合 NTN-地面網路中基於預測性雙連結啟動之切換機制：針對突發遮蔽的魯棒性空間 AI 方法」** 的完整模擬程式碼。

## 專案概述 (Project Overview)

在未來的 5G/6G 混合網路架構中，車輛將優先連接 **地面網路 (Terrestrial Networks, TN)** 以獲取高頻寬與低延遲，並使用 **非地面網路 (NTN/衛星)** 作為訊號覆蓋的備援。然而，在 **山區道路 (Mountainous Terrain)** 行駛時，車輛常面臨 **突發性遮蔽 (Sudden Blockage)**，地面訊號會因地形阻擋而瞬間中斷。

傳統的 **被動式 (Reactive)** 切換機制（例如標準的條件式切換 Standard CHO）無法在訊號中斷前及時切換至衛星，因為物理層的切換流程（包含 T310 + T311 + RRC 重建）大約需要 **3 秒** 的時間，導致通訊服務中斷。

本專案提出一種 **具備魯棒性的空間 AI (Spatial XGBoost)** 方法，其特點如下：
1.  **利用空間情境 (Spatial Context)：** 使用位置與幾何資訊，而非依賴歷史訊號趨勢。
2.  **提前預測：** 能在遮蔽發生前 **5 秒** 做出預測。
3.  **無縫切換：** 觸發 **先連後斷 (Make-Before-Break)** 機制，達成 **零中斷 (Zero/Near-Zero Interruption)** 的目標。
4.  **抗噪聲訓練：** 加入 **雜訊增強 (Noise Augmentation)** 技術，即使在 **15公尺 GPS 誤差** 下仍能準確運作。

## 核心功能 (Key Features)

*   **突發遮蔽模擬：** 建立山區道路環境，模擬訊號「懸崖式」下跌的情境。
*   **雙連結邏輯 (Dual-Connectivity)：** 模擬連線狀態轉移：僅地面 -> 建立衛星連線中 -> 雙連結啟動。
*   **魯棒性測試 (Robustness Test)：** 注入高斯雜訊 (Gaussian Noise) 以模擬峽谷地形中的 GPS 飄移問題。
*   **比較分析 (Comparative Analysis)：**
    *   **Baseline 1:** 標準條件式切換 (Standard CHO) - 被動式，反應不及。
    *   **Baseline 2:** 時序 AI 預測 (Time-Series AI) - 無法預測突發地形變化。
    *   **Proposed:** 空間無線環境地圖 (Spatial REM) - 主動式預測。

## 安裝說明 (Installation)

1.  複製此專案：
    ```bash
    git clone https://github.com/Wang-Chi-Hsien/Predictive-DC-Handover-NTN.git
    cd Predictive-DC-Handover-NTN
    ```

2.  安裝相依套件：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法 (Usage)

執行主模擬程式：

```bash
python main.py
```

程式將執行以下步驟：
1.  生成合成環境數據 (包含訓練集與測試集)。
2.  訓練 Time-Series 與 Spatial AI 模型。
3.  執行移動性模擬 (Mobility Simulation)。
4.  在終端機輸出量化結果 (數據)。
5.  生成並顯示兩張視覺化圖表 (訊號分析與連線狀態)。

## 實驗結果 (Results)

### 量化比較 (模擬條件：15m GPS 誤差)

| 方法 (Method) | 中斷時間 (Interruption Duration) | 狀態 | 備註 |
| :--- | :--- | :--- | :--- |
| **Time-Series AI** | ~9.60 秒 | **失敗** | 無法預測網路拓撲改變 |
| **Standard CHO** | ~1.80 秒 | **失敗** | 被動式機制觸發太晚 |
| **Proposed Spatial** | **~0.60 秒** | **成功** | 達成近乎零中斷 (Near-Zero) |

### 視覺化圖表

*   **Figure 1:** 顯示 **本方法 (藍色星號)** 能在訊號掉落前觸發切換，而其他基準方法則在訊號掉落後才反應。
*   **Figure 2:** 展示本方法在進入遮蔽區之前，就已成功建立衛星連線 (State 2)，確保了服務的連續性 (Service Continuity)。

## 參考文獻 (References)

1.  **S. Mondal et al.**, "Intelligent Handover Orchestration in Beyond 5G and Urban V2X Dual Connectivity Networks: A Deep Reinforcement Learning Approach," *IEEE Access*, 2025.
2.  **3GPP TR 38.811**, "Study on New Radio (NR) to support Non-Terrestrial Networks (NTN)."

---
*Created by Wang Chi-Hsien for the AWS Term Project.*