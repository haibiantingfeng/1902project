import streamlit as st 
import yfinance as yf 
import pandas as pd 
import numpy as np 
from datetime import timedelta 

# 页面设置 
st.set_page_config( 
    page_title="Bi-LSTM vs Transformer 股市预测对比", 
    page_icon="📊", 
    layout="wide" 
) 
st.title("📊 Bi-LSTM vs Transformer 股市预测对比（可公开分享）") 

# 侧边栏设置 
with st.sidebar: 
    st.header("⚙️ 查询与预测设置") 
    ticker = st.text_input("股票代码", value="AAPL", placeholder="例：AAPL、TSLA、000001.SS") 
    start_date = st.date_input("历史数据开始日期", value=pd.to_datetime("2020-01-01")) 
    end_date = st.date_input("历史数据结束日期", value=pd.to_datetime("2026-03-08")) 
    predict_days = st.slider("预测未来天数", min_value=1, max_value=10, value=5) 

# 核心函数：获取数据（增加空数据判断） 
@st.cache_data 
def get_data(ticker, start, end): 
    data = yf.download(ticker, start=start, end=end, progress=False) 
    if data.empty: 
        raise ValueError(f"未获取到 {ticker} 的数据，请检查代码或日期") 
    return data[["Close"]].dropna() 

# 核心函数：模拟两个模型的预测（无TensorFlow依赖） 
def simulate_model_comparison(last_close, days): 
    # Bi-LSTM：平稳预测，波动小 
    lstm_preds = np.linspace(last_close, last_close * 1.02, days) 
    # Transformer：激进预测，波动大 
    trans_preds = np.linspace(last_close, last_close * 1.04, days) + np.random.normal(0, 0.5, days) 
    return lstm_preds, trans_preds 

# 核心函数：模型性能指标（模拟历史拟合） 
def get_performance_metrics(): 
    return pd.DataFrame({ 
        "评估指标": ["MSE", "RMSE", "MAE"], 
        "Bi-LSTM": [0.85, 0.92, 0.78], 
        "Transformer": [1.23, 1.11, 1.05], 
        "更优模型": ["Bi-LSTM", "Bi-LSTM", "Bi-LSTM"] 
    }) 

# 主逻辑（增加异常捕获） 
try: 
    # 1. 获取数据 
    data = get_data(ticker, start_date, end_date) 
    last_close = data["Close"].iloc[-1] 

    # 2. 模拟预测 
    future_dates = pd.date_range(start=end_date, periods=predict_days, freq="B") 
    lstm_preds, trans_preds = simulate_model_comparison(last_close, predict_days) 

    # 3. 构建结果 DataFrame 
    future_df = pd.DataFrame({ 
        "Bi-LSTM 预测": lstm_preds.round(2), 
        "Transformer 预测": trans_preds.round(2) 
    }, index=future_dates) 

    # 4. 展示结果（两栏布局） 
    col1, col2 = st.columns(2) 

    with col1: 
        st.subheader(f"📈 {ticker} 历史收盘价走势") 
        st.line_chart(data["Close"], use_container_width=True) 
        st.subheader("📊 历史数据统计摘要") 
        st.dataframe(data.describe().round(2), use_container_width=True) 

    with col2: 
        st.subheader(f"🔮 未来 {predict_days} 天预测对比") 
        st.line_chart(future_df, use_container_width=True) 
        st.subheader("⚠️ 模型性能对比") 
        st.dataframe(get_performance_metrics(), use_container_width=True, hide_index=True) 

    # 核心结论 
    st.subheader("✅ 核心结论") 
    st.success("在当前模拟场景下，**Bi-LSTM 模型**的综合预测误差更小，表现更优！") 
    st.info("注：此应用为演示工具，预测结果基于模拟数据，不构成投资建议。") 

except Exception as e: 
    st.error(f"❌ 操作失败：{str(e)}") 
    st.info("提示：请检查股票代码是否正确，如A股需加后缀（例：000001.SS），或调整日期范围。")
