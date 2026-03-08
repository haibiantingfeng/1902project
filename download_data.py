import yfinance as yf 
import pandas as pd 

# 定义股票列表 
tickers = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"] 

# 批量下载 
for ticker in tickers: 
    print(f"正在下载 {ticker} 的数据...") 
    data = yf.download(ticker, start="2010-01-01", end="2024-12-31") 
    data.to_csv(f"data/{ticker}_stock_data.csv") 
    print(f"{ticker} 数据已保存到 data/{ticker}_stock_data.csv")
