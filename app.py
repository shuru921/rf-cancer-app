import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

st.set_page_config(page_title="隨機森林乳癌診斷系統", layout="wide", page_icon="🩺")
st.title("🩺 隨機森林：乳癌預測互動系統")
st.caption("資料來源：Breast Cancer Wisconsin Dataset (sklearn 內建)")

# ── 側邊欄：參數設定 ─────────────────────────────────────────
st.sidebar.header("⚙️ 模型參數設定")

n_trees   = st.sidebar.slider("樹的數量 (n_estimators)", 10, 300, 100, step=10)
depth     = st.sidebar.slider("最大深度 (max_depth)", 1, 20, 5)
test_size = st.sidebar.slider("測試集比例 (test_size)", 0.1, 0.4, 0.2, step=0.05)
min_split = st.sidebar.slider("最小分裂樣本數 (min_samples_split)", 2, 20, 2)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("📂 上傳自訂 CSV（可選）", type=["csv"])

# ── 載入資料 ─────────────────────────────────────────────────
@st.cache_data
def load_default():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["diagnosis"] = df["target"].map({1: "良性 Benign", 0: "惡性 Malignant"})
    return data, df

data, df = load_default()

if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded)
        st.sidebar.success(f"已載入 {len(user_df)} 筆資料，將使用預設資料集進行示範。")
    except Exception as e:
        st.sidebar.error(f"讀取失敗：{e}")

# ── 訓練模型 ─────────────────────────────────────────────────
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

@st.cache_resource
def train(n, d, ms, ts):
    clf = RandomForestClassifier(
        n_estimators=n, max_depth=d,
        min_samples_split=ms, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

clf    = train(n_trees, depth, min_split, test_size)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, y_pred)

# ── Tab 佈局 ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 數據總覽", "🎯 模型表現", "📈 特徵重要性", "🔍 即時預測"])

# ── Tab 1：數據總覽 ───────────────────────────────────────────
with tab1:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("總樣本數", len(df))
    col_b.metric("良性 (Benign)", int(df["target"].sum()))
    col_c.metric("惡性 (Malignant)", int((df["target"] == 0).sum()))

    st.subheader("原始數據（前 10 筆）")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("類別分佈")
    pie = px.pie(df, names="diagnosis", color_discrete_sequence=["#2196F3", "#F44336"])
    st.plotly_chart(pie, use_container_width=True)

# ── Tab 2：模型表現 ───────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("準確度 Accuracy", f"{acc:.2%}")
        report = classification_report(y_test, y_pred, target_names=["惡性", "良性"], output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with col2:
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        labels = ["惡性 (0)", "良性 (1)"]
        fig_cm = ff.create_annotated_heatmap(
            cm, x=labels, y=labels,
            colorscale="Blues", showscale=True
        )
        fig_cm.update_layout(
            title="混淆矩陣 Confusion Matrix",
            xaxis_title="預測值", yaxis_title="實際值"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC 曲線
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f"ROC 曲線 (AUC = {roc_auc:.3f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        color_discrete_sequence=["#2196F3"]
    )
    fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

# ── Tab 3：特徵重要性 ─────────────────────────────────────────
with tab3:
    st.subheader("前 15 大關鍵診斷指標")
    importances = pd.DataFrame({
        "feature": data.feature_names,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    fig_imp = px.bar(
        importances, x="importance", y="feature",
        orientation="h", color="importance",
        color_continuous_scale="Blues",
        title="特徵重要性 Feature Importance"
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("特徵相關性熱圖（前 10 特徵）")
    top10 = importances["feature"].head(10).tolist()
    corr = df[top10].corr()
    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="特徵相關矩陣")
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 4：即時預測 ───────────────────────────────────────────
with tab4:
    st.subheader("輸入生理指標進行即時診斷")
    st.info("以下僅展示前 10 個最重要的特徵供輸入，其餘特徵使用訓練集平均值補齊。")

    top_features = importances["feature"].head(10).tolist()
    input_vals   = {}

    cols = st.columns(2)
    for i, feat in enumerate(top_features):
        col_idx = i % 2
        feat_idx = list(data.feature_names).index(feat)
        mn  = float(df[feat].min())
        mx  = float(df[feat].max())
        avg = float(df[feat].mean())
        input_vals[feat] = cols[col_idx].number_input(feat, min_value=mn, max_value=mx, value=avg, format="%.4f")

    # 組合完整輸入向量
    full_input = []
    for feat in data.feature_names:
        if feat in input_vals:
            full_input.append(input_vals[feat])
        else:
            full_input.append(float(df[feat].mean()))

    if st.button("🔬 開始預測", type="primary"):
        pred      = clf.predict([full_input])[0]
        prob      = clf.predict_proba([full_input])[0]
        label     = "良性 Benign" if pred == 1 else "惡性 Malignant"
        confidence = prob[pred]

        if pred == 1:
            st.success(f"預測結果：**{label}**　信心度：{confidence:.1%}")
        else:
            st.error(f"預測結果：**{label}**　信心度：{confidence:.1%}")

        prob_df = pd.DataFrame({
            "類別": ["惡性 Malignant", "良性 Benign"],
            "機率": [prob[0], prob[1]]
        })
        fig_prob = px.bar(prob_df, x="類別", y="機率", color="類別",
                          color_discrete_sequence=["#F44336", "#2196F3"],
                          title="預測機率分佈", range_y=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

        st.caption("⚠️ 本系統僅供學術示範，不作為臨床診斷依據。")
