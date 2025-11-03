# LSTM 外汇交易系统

<div align="center">

**基于深度学习的外汇自动交易系统**

完整的工作流程：数据获取 → 模型训练 → 高性能回测 → 实盘交易

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 系统概览

这是一个完整的外汇交易系统，使用LSTM神经网络预测EURUSD价格走势，并提供：

- ✅ **智能信号生成**：基于60根H1 K线的LSTM预测
- ✅ **高性能回测**：信号预计算技术，速度提升50-100倍
- ✅ **风险管理**：固定手数、佣金模拟
- ✅ **实盘交易**：与MT5无缝集成的自动交易
- ✅ **可视化报告**：交互式HTML回测报告

---

## 🚀 核心特性

### 1. LSTM深度学习模型
- **架构**: 2层LSTM（隐藏层100单元）+ 全连接层
- **输入**: 60个H1 K线的OHLC数据
- **输出**: BUY / SELL / HOLD 三分类信号
- **训练**: 10年历史数据，80/20训练测试集

### 2. 高性能回测系统（重要创新）⭐

**传统回测的问题**:
```python
# 每个bar都调用模型 → 10年数据约87,600次调用 → 非常慢！
def next(self):
    sequence = get_last_60_bars()
    signal = model.predict(sequence)  # 重复调用
    execute_trade(signal)
```

**我们的优化方案**:
```python
# 回测前一次性批量计算所有信号 → 只调用1次 → 超快！
signals = precompute_signals(all_data)  # 批量预测
# 回测时直接查表
def next(self):
    signal = signals[current_index]  # 查表，无模型调用
    execute_trade(signal)
```

**性能对比**:
- **传统方法**: 30-60分钟（87,600次模型调用）
- **优化方法**: 1-2分钟（1次批量预测 + 查表）
- **速度提升**: 50-100倍 🚀

### 3. 风险管理系统
- 固定手数交易（0.01手）
- 交易佣金模拟（0.02%）
- 初始资金配置（10万单位）

---

## 🛠️ 快速开始

### 前置要求

```bash
# 1. Python 3.8+
python --version

# 2. 安装依赖
pip install MetaTrader5 torch pandas numpy scikit-learn backtesting

# 3. 安装并登录 MetaTrader 5（建议使用模拟账户）
```

### 两步上手

```bash
# 步骤1: 训练模型并回测（首次必须）
python lstm_forex_backtest.py

# 步骤2: 实盘交易（谨慎使用，先看 LIVE_TRADING_README.md）
python live_trading.py
```

---

## 🏗️ 系统架构

### 文件结构

```
python-lstm-forex-backtest/
│
├── 📊 核心脚本
│   ├── lstm_forex_backtest.py          # [1] 模型训练 + 高性能回测 ⭐
│   └── live_trading.py                 # [2] 实盘自动交易 ⚠️
│
├── 🤖 模型文件（自动生成）
│   ├── lstm_model.pth                  # 训练好的LSTM模型
│   └── scaler.pkl                      # 数据标准化器
│
├── 📈 回测报告（自动生成）
│   ├── backtest_report.html            # 交互式可视化报告
│   └── backtest_report.txt             # 文本统计报告
│
└── 📖 文档
    ├── README.md                       # 项目说明（本文件）
    ├── prompt.md                       # 开发规范文档
    └── LIVE_TRADING_README.md          # 实盘交易使用指南
```

---

## ⚡ 性能优化详解

### 优化前后对比

| 指标 | 传统方法 | 优化方法 | 提升 |
|-----|---------|---------|------|
| **模型调用次数** | ~87,600次 | 1次（批量） | 99.999%减少 |
| **回测耗时** | 30-60分钟 | 1-2分钟 | 50-100倍 |
| **内存占用** | 不稳定 | 稳定 | 更优 |

---

## ⚙️ 配置参数

### LSTM模型参数

```python
SEQUENCE_LENGTH = 60      # 输入序列长度（60根H1 K线）
HIDDEN_SIZE = 100         # LSTM隐藏层大小
NUM_LAYERS = 2            # LSTM层数
BATCH_SIZE = 64           # 训练批大小
NUM_EPOCHS = 20           # 训练轮数
LEARNING_RATE = 0.001     # 学习率
```

### 回测参数

```python
INITIAL_CASH = 100000     # 初始资金
COMMISSION = 0.0002       # 交易佣金（0.02%）
SPLIT_RATIO = 0.8         # 训练集比例（80%）
```

---

## ❓ 常见问题

### Q1: 回测结果可靠吗？

**A**: 回测基于历史数据，存在以下局限：
- ✅ 可以评估策略的历史表现
- ✅ 可以测试不同参数组合
- ❌ 不能保证未来表现
- ❌ 可能存在过拟合风险

### Q2: 可以用于其他货币对吗？

**A**: 可以，修改代码中的 `symbol = "EURUSD"` 为其他货币对（如 "GBPUSD"）。但需要重新训练模型。

### Q3: 回测太慢怎么办？

**A**: 
- ✅ 确保使用了优化后的代码（带 `precompute_signals`）
- ✅ 检查是否有GPU可用（训练阶段）
- ✅ 减少回测年限（如5年 `days=365*5`）
- ❌ 不要在 `Strategy.next()` 中调用模型！

### Q4: 实盘交易安全吗？

**A**: 
- ⚠️ 外汇交易存在高风险
- ⚠️ 算法交易可能快速亏损
- ⚠️ 必须在模拟账户充分测试
- ⚠️ 使用真实资金前请咨询专业人士

---

## 📖 扩展阅读

- **LSTM原理**: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **回测方法论**: [Backtesting.py Documentation](https://kernc.github.io/backtesting.py/)
- **MT5 API**: [MetaTrader 5 Python Integration](https://www.mql5.com/en/docs/python_metatrader5)

---

## ⚠️ 免责声明

**重要提示**:

1. **教育目的**: 本系统仅供学习和研究使用
2. **高风险警告**: 外汇交易存在极高风险，可能导致全部本金损失
3. **无保证**: 历史表现不代表未来结果
4. **自担风险**: 使用本系统进行实盘交易的所有风险和责任由用户自行承担
5. **专业建议**: 投资前请咨询专业的金融顾问

**使用本系统即表示您已阅读并同意上述免责声明。**

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！**

Made with ❤️ for LSTM Traders

</div>