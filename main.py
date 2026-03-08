import streamlit as st 
import yfinance as yf 
import pandas as pd 

# 页面基础设置 
st.set_page_config(page_title="股票数据查询工具", page_icon="📈") 
st.title("📈 股票历史数据查询工具") 

# 侧边栏输入区 
with st.sidebar: 
    st.header("查询设置") 
    ticker = st.text_input("股票代码", value="AAPL", placeholder="例：AAPL、TSLA、000001.SS") 
    start_date = st.date_input("开始日期", value=pd.to_datetime("2020-01-01")) 
    end_date = st.date_input("结束日期", value=pd.to_datetime("2026-03-08")) 

# 核心查询逻辑 
if st.button("执行查询", type="primary"): 
    with st.spinner("正在从 Yahoo Finance 获取数据..."): 
        try: 
            # 下载数据 
            data = yf.download(ticker, start=start_date, end=end_date) 
            # 展示结果 
            st.subheader(f"{ticker} 收盘价走势") 
            st.line_chart(data["Close"], use_container_width=True) 
            st.subheader("最新10条数据预览") 
            st.dataframe(data.tail(10), use_container_width=True) 
            st.success(f"✅ 成功获取 {ticker} 共 {len(data)} 条数据！") 
        except Exception as e: 
            st.error(f"❌ 查询失败：{str(e)}") 
            st.info("提示：请检查股票代码是否正确，如A股需加后缀（例：000001.SS）")
