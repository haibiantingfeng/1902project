import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
import os

# ========== 1. 下载数据：5只股票、5年（2020–2025） ==========
print("正在下载数据...")
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
start = "2020-01-01"
end = "2025-01-01"

# 下载完整数据（不只是Close）
data = yf.download(tickers, start=start, end=end)
print(f"下载完成，数据形状：{data.shape}")

# ========== 2. 特征工程 ==========
def add_technical_indicators(df):
    df = df.copy()
    
    # 移动平均线
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    
    # RSI (相对强弱指标)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 波动率
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 价格变化率
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    return df

# 对每只股票添加技术指标
features_list = []
for ticker in tickers:
    ticker_data = pd.DataFrame({
        'Open': data['Open'][ticker],
        'High': data['High'][ticker],
        'Low': data['Low'][ticker],
        'Close': data['Close'][ticker],
        'Volume': data['Volume'][ticker]
    })
    ticker_data = add_technical_indicators(ticker_data)
    features_list.append(ticker_data)

# 合并所有股票的特征
all_features = pd.concat(features_list, axis=1)
all_features = all_features.dropna()

print(f"特征工程完成，特征数量：{all_features.shape[1]}")

# ========== 3. 数据预处理 ==========
scaler = MinMaxScaler()
scaled = scaler.fit_transform(all_features)

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)
print(f"数据形状：X={X.shape}, y={y.shape}")

# ========== 4. 构建改进的 BiLSTM 模型 ==========
def build_bilstm(input_shape, output_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ========== 5. 构建改进的 Transformer 模型 ==========
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)
    
    x = Dense(ff_dim, activation='gelu')(x)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    return x

def build_transformer(input_shape, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=4):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(y.shape[1])(x)
    
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ========== 6. 回调函数 ==========
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# ========== 7. 开始训练 ==========
start_time = time.time()

# 训练 BiLSTM
print("\n=== 训练 BiLSTM ===")
bilstm = build_bilstm(X.shape[1:], y.shape[1])
bilstm.summary()

bilstm_history = bilstm.fit(
    X, y,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 保存 BiLSTM 模型
os.makedirs('models', exist_ok=True)
bilstm.save("models/bilstm_5stock_5y_improved.h5")
print("BiLSTM 模型已保存")

# 训练 Transformer
print("\n=== 训练 Transformer ===")
transformer = build_transformer(X.shape[1:])
transformer.summary()

transformer_history = transformer.fit(
    X, y,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 保存 Transformer 模型
transformer.save("models/transformer_5stock_5y_improved.h5")
print("Transformer 模型已保存")

end_time = time.time()
total_time = (end_time - start_time) / 60

print(f"\n{'='*50}")
print(f"总训练时间：{round(total_time, 1)} 分钟")
print(f"{'='*50}")

# ========== 8. 评估模型 ==========
print("\n=== 模型评估 ===")
bilstm_loss, bilstm_mae = bilstm.evaluate(X, y, verbose=0)
transformer_loss, transformer_mae = transformer.evaluate(X, y, verbose=0)

print(f"BiLSTM - Loss: {bilstm_loss:.6f}, MAE: {bilstm_mae:.6f}")
print(f"Transformer - Loss: {transformer_loss:.6f}, MAE: {transformer_mae:.6f}")

if bilstm_mae < transformer_mae:
    print(f"\n✅ BiLSTM 表现更好（MAE 更低）")
else:
    print(f"\n✅ Transformer 表现更好（MAE 更低）")

print("\n模型已保存到 models/ 目录，可上传 GitHub！")
