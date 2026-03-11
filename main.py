import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report 

# ---------------------- 
# 页面配置 
# ---------------------- 
st.set_page_config(page_title="双模型训练演示", layout="wide") 
st.title("📊 双模型训练系统（Logistic + RandomForest）") 

# ---------------------- 
# 1. 数据准备 
# ---------------------- 
st.subheader("1. 数据加载") 

data_option = st.radio("数据来源", ["生成模拟数据", "上传CSV文件"]) 
df = None 

if data_option == "生成模拟数据": 
    st.info("生成二分类模拟数据") 
    n_samples = st.slider("样本数量", 500, 5000, 1000) 
    X = np.random.randn(n_samples, 10) 
    y = np.random.randint(0, 2, size=n_samples) 
    df = pd.DataFrame(X) 
    df["target"] = y 
    st.dataframe(df.head(5)) 

else: 
    uploaded = st.file_uploader("上传CSV（最后一列是标签）", type="csv") 
    if uploaded: 
        df = pd.read_csv(uploaded) 
        st.dataframe(df.head(5)) 

# ---------------------- 
# 2. 训练准备 
# ---------------------- 
if df is not None: 
    st.subheader("2. 训练设置") 

    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2) 
    random_state = st.number_input("随机种子", value=42) 

    # 划分 X y 
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1] 

    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=test_size, random_state=random_state 
    ) 

    st.write(f"训练集: {X_train.shape[0]} 条") 
    st.write(f"测试集: {X_test.shape[0]} 条") 

    # ---------------------- 
    # 3. 训练两个模型 
    # ---------------------- 
    st.subheader("3. 开始训练") 
    if st.button("🚀 训练 Logistic + RandomForest"): 

        with st.spinner("训练中..."): 
            # 模型1 
            lr = LogisticRegression(max_iter=1000) 
            lr.fit(X_train, y_train) 
            y_pred_lr = lr.predict(X_test) 
            acc_lr = accuracy_score(y_test, y_pred_lr) 

            # 模型2 
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state) 
            rf.fit(X_train, y_train) 
            y_pred_rf = rf.predict(X_test) 
            acc_rf = accuracy_score(y_test, y_pred_rf) 

        # 展示结果 
        st.success("训练完成！") 
        col1, col2 = st.columns(2) 
        with col1: 
            st.metric("Logistic Regression 准确率", f"{acc_lr:.2%}") 
            st.text(classification_report(y_test, y_pred_lr)) 
        with col2: 
            st.metric("Random Forest 准确率", f"{acc_rf:.2%}") 
            st.text(classification_report(y_test, y_pred_rf)) 

        # 画对比图 
        st.subheader("4. 模型对比图") 
        models = ["Logistic", "RandomForest"] 
        accs = [acc_lr, acc_rf] 
        fig, ax = plt.subplots() 
        ax.bar(models, accs, color=["#1f77b4", "#ff7f0e"]) 
        ax.set_ylim(0, 1) 
        ax.set_ylabel("准确率") 
        ax.set_title("双模型表现对比") 
        st.pyplot(fig) 

else: 
    st.warning("请先加载数据")
