import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, MultiHeadAttention, LayerNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# 设置页面
st.set_page_config(page_title="股票模型比较工具", page_icon="📈", layout="wide")
st.title("📈 股票模型比较：Bi-LSTM vs Transformer")

# 侧边栏输入
with st.sidebar:
    st.header("参数设置")
    ticker = st.text_input("股票代码", value="AAPL")
    start_date = st.date_input("开始日期", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("结束日期", value=pd.to_datetime("2024-12-31"))
    time_steps = st.slider("时间步长", 5, 30, 10)
    epochs = st.slider("训练轮数", 10, 100, 50)

# 下载数据
@st.cache_data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    # 计算技术指标
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = ta.momentum.rsi(data['Close'])
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data = data.dropna()
    return data

# 数据预处理
@st.cache_data
def preprocess_data(data, time_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# 构建 Bi-LSTM 模型
def build_bi_lstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 构建 Transformer 模型
def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 计算性能指标
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

# 主逻辑
if st.button("运行分析", type="primary"):
    with st.spinner("正在处理数据..."):
        # 获取数据
        data = get_data(ticker, start_date, end_date)
        
        # 预处理数据
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, time_steps)
        
        # 构建和训练模型
        st.subheader("模型训练")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🔄 训练 Bi-LSTM 模型...")
            bi_lstm = build_bi_lstm((time_steps, 1))
            bi_lstm.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            st.write("✅ Bi-LSTM 模型训练完成")
        
        with col2:
            st.write("🔄 训练 Transformer 模型...")
            transformer = build_transformer((time_steps, 1))
            transformer.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            st.write("✅ Transformer 模型训练完成")
        
        # 预测
        bi_lstm_pred = bi_lstm.predict(X_test)
        transformer_pred = transformer.predict(X_test)
        
        # 反归一化
        bi_lstm_pred = scaler.inverse_transform(bi_lstm_pred)
        transformer_pred = scaler.inverse_transform(transformer_pred)
        y_test_actual = scaler.inverse_transform(y_test)
        
        # 计算指标
        bi_lstm_metrics = calculate_metrics(y_test_actual, bi_lstm_pred)
        transformer_metrics = calculate_metrics(y_test_actual, transformer_pred)
        
        # 可视化
        st.subheader("📊 数据和指标")
        
        # OHLC 图表
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, 
                           row_heights=[0.5, 0.25, 0.25])
        
        # OHLC + MA
        fig.add_trace(go.Candlestick(x=data.index, 
                                     open=data['Open'], 
                                     high=data['High'], 
                                     low=data['Low'], 
                                     close=data['Close'], 
                                     name='价格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MA5'], name='MA5', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='red')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='green')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
        
        fig.update_layout(height=600, title=f"{ticker} 技术指标分析")
        st.plotly_chart(fig, use_container_width=True)
        
        # 预测图表
        st.subheader("📈 模型预测对比")
        pred_dates = data.index[-len(y_test_actual):]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=y_test_actual.flatten(), name='实际价格', line=dict(color='black')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=bi_lstm_pred.flatten(), name='Bi-LSTM 预测', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=transformer_pred.flatten(), name='Transformer 预测', line=dict(color='red')))
        fig_pred.update_layout(height=400, title=f"{ticker} 价格预测对比")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # 性能指标
        st.subheader("📋 模型性能指标")
        metrics_df = pd.DataFrame({
            '指标': ['MSE', 'RMSE', 'MAE'],
            'Bi-LSTM': bi_lstm_metrics,
            'Transformer': transformer_metrics
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        # 模型比较
        st.subheader("🏆 模型比较结果")
        
        # 计算平均排名
        bi_lstm_rank = sum([1 if bi_lstm_metrics[i] < transformer_metrics[i] else 0 for i in range(3)])
        transformer_rank = sum([1 if transformer_metrics[i] < bi_lstm_metrics[i] else 0 for i in range(3)])
        
        if bi_lstm_rank > transformer_rank:
            st.success("✅ Bi-LSTM 模型表现更好！")
        elif transformer_rank > bi_lstm_rank:
            st.success("✅ Transformer 模型表现更好！")
        else:
            st.info("📊 两个模型表现相当。")
        
        # 数据预览
        st.subheader("📄 数据预览")
        st.dataframe(data.tail(10), use_container_width=True)
