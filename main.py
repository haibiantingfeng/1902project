import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Bidirectional 
import joblib 

# ---------------------- 页面标题 ---------------------- 
st.title("股市方向预测：Bidirectional LSTM vs Transformer") 
st.subheader("AIE 课程项目 - 简易版") 

# ---------------------- 模拟模型训练（仅演示） ---------------------- 
@st.cache_resource 
def train_dummy_model(): 
    # 模拟数据 
    X = np.random.rand(1000, 10, 5)  # 1000个样本，每个样本10个时间步，5个特征 
    y = np.random.randint(0, 2, size=1000)  # 0=跌，1=涨 
    
    # 构建简易双向LSTM模型 
    model = Sequential([ 
        Bidirectional(LSTM(32, input_shape=(10, 5))), 
        Dense(16, activation='relu'), 
        Dense(1, activation='sigmoid') 
    ]) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    
    # 训练（仅5轮，快速演示） 
    model.fit(X, y, epochs=5, batch_size=32, verbose=0) 
    
    # 计算准确率 
    _, acc = model.evaluate(X, y, verbose=0) 
    return model, acc 

model, accuracy = train_dummy_model() 

# ---------------------- 显示模型信息 ---------------------- 
st.write(f"当前模型准确率：{accuracy:.2%}") 

# ---------------------- 输入区 ---------------------- 
st.subheader("输入股票数据（模拟）") 
col1, col2, col3 = st.columns(3) 
with col1: 
    open_price = st.number_input("开盘价", min_value=0.0, value=100.0) 
with col2: 
    close_price = st.number_input("收盘价", min_value=0.0, value=102.0) 
with col3: 
    volume = st.number_input("成交量", min_value=0.0, value=1e6) 

# 构造输入序列（模拟10个时间步） 
input_data = np.array([[open_price, close_price, volume, open_price, close_price] for _ in range(10)]) 
input_data = input_data.reshape(1, 10, 5)  # 适配模型输入形状 

# ---------------------- 预测按钮 ---------------------- 
if st.button("预测股市方向"): 
    pred = model.predict(input_data, verbose=0)[0][0] 
    direction = "📈 上涨" if pred > 0.5 else "📉 下跌" 
    st.success(f"预测结果：{direction}（置信度：{pred:.2%}）")
