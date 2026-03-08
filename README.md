# 股票市场方向预测模型比较

## 项目简介

本项目使用 Python 和 Streamlit 构建了一个交互式 Web 应用，用于比较双向 LSTM 和 Transformer 两种机器学习模型在预测股票市场方向上的表现。用户可以输入股票代码，系统会自动从 Yahoo Finance 下载历史数据，进行预处理和特征工程，然后使用两种模型进行预测并展示结果。

## 功能特点

- 支持用户输入股票代码（如 AAPL, TSLA）
- 自动从 Yahoo Finance 下载历史股价数据
- 进行数据预处理和特征工程（计算 MA、RSI、MACD 等技术指标）
- 加载并运行双向 LSTM 和 Transformer 模型进行股市方向预测
- 可视化展示：历史价格走势、技术指标、两种模型的预测结果对比
- 显示预测准确率和关键指标

## 项目结构

```
.
├── main.py              # 主程序
├── requirements.txt     # 依赖包
├── README.md            # 项目说明
├── models/              # 训练好的模型文件
│   ├── bi_lstm.h5       # 双向 LSTM 模型
│   └── transformer.h5   # Transformer 模型
└── data/                # 数据目录
```

## 安装步骤

1. 克隆项目到本地
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行应用：
   ```bash
   streamlit run main.py
   ```

## 使用方法

1. 在浏览器中打开应用
2. 在侧边栏输入股票代码（如 AAPL, TSLA）
3. 点击"预测"按钮
4. 查看结果：
   - 历史价格走势
   - 技术指标（MA、RSI、MACD）
   - 两种模型的预测结果对比
   - 预测准确率和关键指标

## 模型说明

- **双向 LSTM 模型**：利用长短期记忆网络的双向特性，捕获股票价格的时序依赖关系
- **Transformer 模型**：使用自注意力机制，能够更好地捕获长期依赖关系

## 部署指南

### 部署到 Streamlit Community Cloud

1. 将项目推送到 GitHub 仓库
2. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 登录并点击"New app"
4. 选择你的 GitHub 仓库
5. 选择主分支和 main.py 文件
6. 点击"Deploy"

### 推送到 GitHub 仓库

```bash
# 初始化 git 仓库
git init

# 添加文件
git add .

# 提交更改
git commit -m "Initial commit"

# 关联到远程仓库
git remote add origin https://github.com/haibiantingfeng/1902project.git

# 推送代码
git push -u origin main
```

## 注意事项

- 本项目仅用于教育和研究目的，不构成投资建议
- 股票市场预测存在风险，预测结果仅供参考
- 首次运行时会自动训练模型，可能需要一些时间
- 模型性能可能因不同股票和市场环境而有所差异