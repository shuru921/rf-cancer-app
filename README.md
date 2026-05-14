# 🩺 隨機森林乳癌預測系統

一款以 **Random Forest（隨機森林）** 為核心的互動式醫療診斷輔助 App，使用 Streamlit 框架開發，部署於 Streamlit Community Cloud。

🔗 **線上體驗：** [https://rf-cancer-app-xulgappcn2rpbwjadbbsbvh.streamlit.app/](https://rf-cancer-app-xulgappcn2rpbwjadbbsbvh.streamlit.app/)

---

## 專案簡介

本專案將機器學習中的集成學習方法（Ensemble Learning）應用於臨床資訊工程領域，以 **乳癌威斯康辛資料集（Breast Cancer Wisconsin Dataset）** 為例，展示如何透過調整模型超參數、視覺化診斷結果，輔助臨床決策判斷。

---

## 演算法與資料集

| 項目 | 內容 |
|------|------|
| 演算法 | Random Forest（隨機森林）— 監督式學習／集成學習 |
| 資料集 | Breast Cancer Wisconsin Dataset（sklearn 內建） |
| 樣本數 | 569 筆（良性 357 筆、惡性 212 筆） |
| 特徵數 | 30 個生理指標（細胞半徑、紋理、對稱性等） |
| 任務類型 | 二元分類（良性 / 惡性）|

---

## 功能列表

- **超參數調整**：透過側邊欄即時調整 `n_estimators`、`max_depth`、`test_size`、`min_samples_split`
- **數據總覽**：樣本統計、原始數據預覽、類別分佈圓餅圖
- **模型表現**：準確度（Accuracy）、分類報告、混淆矩陣（Confusion Matrix）、ROC 曲線與 AUC
- **特徵重要性**：前 15 大關鍵指標長條圖、特徵相關性熱圖
- **即時預測**：輸入生理指標數值，即時回傳診斷結果與預測機率分佈

---

## 本地執行方式

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 啟動 App
streamlit run app.py
```

瀏覽器自動開啟 `http://localhost:8501`

---

## 技術堆疊

- **Python 3.10+**
- **Streamlit** — 互動式 App 框架
- **scikit-learn** — Random Forest 模型訓練與評估
- **Plotly** — 互動式圖表視覺化
- **Pandas / NumPy** — 數據處理

---

> ⚠️ 本系統僅供學術示範，不作為臨床診斷依據。
