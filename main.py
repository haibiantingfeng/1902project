import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
import os
from ta import add_all_ta_features

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 缓存数据
def get_stock_data(ticker, period="1y"):
    """从 Yahoo Finance 下载股票数据"""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# 数据预处理和特征工程
def preprocess_data(data):
    """数据预处理和特征工程"""
    # 计算技术指标
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    
    # 创建目标变量：如果下一天的收盘价高于今天，则为1，否则为0
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    
    # 移除 NaN 值
    data = data.dropna()
    
    # 选择特征列
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                      'volume_adi', 'volume_obv', 'volume_cmf',
                      'volatility_atr', 'trend_macd', 'trend_macd_signal',
                      'trend_rsi', 'momentum_rsi', 'trend_sma_20', 'trend_sma_50']
    
    # 分离特征和目标
    X = data[feature_columns].values
    y = data['Target'].values
    
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns, data

# 创建时间序列数据
def create_sequences(X, y, time_steps=10):
    """创建时间序列数据"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

# 构建双向 LSTM 模型
def build_bi_lstm_model(input_shape):
    """构建双向 LSTM 模型"""
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 构建 Transformer 模型
def build_transformer_model(input_shape):
    """构建 Transformer 模型"""
    inputs = Input(shape=input_shape)
    
    # 多头注意力层
    x = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    
    # 全局池化
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # 输出层
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练和保存模型
def train_and_save_models(X_train, y_train, model_dir):
    """训练和保存模型"""
    # 构建双向 LSTM 模型
    bi_lstm_model = build_bi_lstm_model(X_train.shape[1:])
    bi_lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    bi_lstm_model.save(os.path.join(model_dir, 'bi_lstm.h5'))
    
    # 构建 Transformer 模型
    transformer_model = build_transformer_model(X_train.shape[1:])
    transformer_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    transformer_model.save(os.path.join(model_dir, 'transformer.h5'))
    
    return bi_lstm_model, transformer_model

# 加载模型
def load_models(model_dir):
    """加载模型"""
    bi_lstm_path = os.path.join(model_dir, 'bi_lstm.h5')
    transformer_path = os.path.join(model_dir, 'transformer.h5')
    
    # 如果模型文件不存在，返回 None
    if not os.path.exists(bi_lstm_path) or not os.path.exists(transformer_path):
        return None, None
    
    bi_lstm_model = load_model(bi_lstm_path)
    transformer_model = load_model(transformer_path)
    return bi_lstm_model, transformer_model

# 预测函数
def predict_market_direction(models, X_test):
    """预测市场方向"""
    bi_lstm_model, transformer_model = models
    
    # 双向 LSTM 预测
    bi_lstm_pred = bi_lstm_model.predict(X_test)
    bi_lstm_pred = (bi_lstm_pred > 0.5).astype(int)
    
    # Transformer 预测
    transformer_pred = transformer_model.predict(X_test)
    transformer_pred = (transformer_pred > 0.5).astype(int)
    
    return bi_lstm_pred, transformer_pred

# 可视化函数
def plot_results(data, bi_lstm_pred, transformer_pred, y_test):
    """可视化结果"""
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 历史价格走势
    ax1.plot(data.index[-len(y_test):], data['Close'].iloc[-len(y_test):], label='实际价格')
    ax1.set_title('历史价格走势')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格')
    ax1.legend()
    
    # 技术指标：RSI
    ax2.plot(data.index[-len(y_test):], data['trend_rsi'].iloc[-len(y_test):], label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--', label='超买')
    ax2.axhline(y=30, color='g', linestyle='--', label='超卖')
    ax2.set_title('相对强弱指标 (RSI)')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('RSI')
    ax2.legend()
    
    # 技术指标：MACD
    ax3.plot(data.index[-len(y_test):], data['trend_macd'].iloc[-len(y_test):], label='MACD')
    ax3.plot(data.index[-len(y_test):], data['trend_macd_signal'].iloc[-len(y_test):], label='信号线')
    ax3.set_title('MACD 指标')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('MACD')
    ax3.legend()
    
    # 预测结果对比
    ax4.plot(data.index[-len(y_test):], y_test, label='实际方向', marker='o', linestyle='')
    ax4.plot(data.index[-len(y_test):], bi_lstm_pred, label='双向 LSTM 预测', marker='x', linestyle='')
    ax4.plot(data.index[-len(y_test):], transformer_pred, label='Transformer 预测', marker='s', linestyle='')
    ax4.set_title('模型预测结果对比')
    ax4.set_xlabel('日期')
    ax4.set_ylabel('方向 (1=上涨, 0=下跌)')
    ax4.legend()
    
    plt.tight_layout()
    return fig

# 主函数
def main():
    st.title('股票市场方向预测模型比较')
    st.sidebar.title('参数设置')
    
    # 用户输入股票代码
    ticker = st.sidebar.text_input('股票代码', 'AAPL')
    period = st.sidebar.selectbox('时间范围', ['1y', '2y', '5y', '10y'], index=0)
    time_steps = st.sidebar.slider('时间步长', 5, 30, 10)
    
    # 模型目录
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 下载数据
    data = get_stock_data(ticker, period)
    
    # 数据预处理和特征工程
    X, y, scaler, feature_columns, processed_data = preprocess_data(data)
    
    # 创建时间序列数据
    X_seq, y_seq = create_sequences(X, y, time_steps)
    
    # 划分训练集和测试集
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # 加载或训练模型
    bi_lstm_model, transformer_model = load_models(model_dir)
    
    if bi_lstm_model is None or transformer_model is None:
        st.info('模型文件不存在，正在训练模型...')
        bi_lstm_model, transformer_model = train_and_save_models(X_train, y_train, model_dir)
        st.success('模型训练完成并保存')
    
    # 预测
    bi_lstm_pred, transformer_pred = predict_market_direction((bi_lstm_model, transformer_model), X_test)
    
    # 计算准确率
    bi_lstm_accuracy = accuracy_score(y_test, bi_lstm_pred)
    transformer_accuracy = accuracy_score(y_test, transformer_pred)
    
    # 显示结果
    st.header('预测结果')
    
    # 准确率对比
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('双向 LSTM 模型')
        st.metric('准确率', f"{bi_lstm_accuracy:.2f}")
    with col2:
        st.subheader('Transformer 模型')
        st.metric('准确率', f"{transformer_accuracy:.2f}")
    
    # 分类报告
    st.subheader('分类报告')
    col1, col2 = st.columns(2)
    with col1:
        st.text('双向 LSTM 模型:')
        st.text(classification_report(y_test, bi_lstm_pred))
    with col2:
        st.text('Transformer 模型:')
        st.text(classification_report(y_test, transformer_pred))
    
    # 可视化结果
    st.subheader('可视化分析')
    fig = plot_results(processed_data, bi_lstm_pred, transformer_pred, y_test)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
