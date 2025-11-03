# LSTM实盘交易系统 - 使用指南

## ⚠️ 重要警告

**这是实盘交易程序，会进行真实交易！**

- ✅ **必须先在MT5模拟账户测试至少1个月**
- ✅ **从最小手数开始（0.01手）**
- ✅ **确保已运行双时间框架回测验证策略表现**
- ⚠️ **交易有风险，可能导致本金全部损失**
- ⚠️ **作者不对任何交易损失负责**

## 📌 使用前必读

在运行实盘交易之前，请确保：

1. ✅ 已运行 `lstm_dual_timeframe_backtest.py` 并验证回测结果
2. ✅ 回测的盈亏比、胜率、最大回撤等指标符合预期
3. ✅ 已在MT5模拟账户测试至少1个月
4. ✅ 完全理解系统的交易逻辑和风险管理机制
5. ✅ 已准备好承担可能的亏损

---

## 📋 系统概述

### 功能特性

1. **自动交易**
   - 实时获取MT5市场数据
   - 使用训练好的LSTM模型生成交易信号
   - 自动执行买入/卖出/平仓操作

2. **风险管理**
   - 自动止损止盈
   - 每日交易次数限制
   - 最大持仓数量控制
   - 滑点保护

3. **监控和日志**
   - 实时显示账户状态
   - 详细的交易日志
   - 持仓情况监控
   - 盈亏统计

4. **安全保护**
   - 交易时间控制
   - 信号置信度过滤
   - 手动紧急停止
   - 退出时可选平仓

---

## 🚀 快速开始

### 前置要求

1. **MT5账户**
   - 已安装MetaTrader 5
   - 已登录账户（建议使用模拟账户）
   - EURUSD交易品种可用

2. **已训练模型**
   - `lstm_model.pth` - 训练好的LSTM模型
   - `scaler.pkl` - 数据预处理器

3. **Python环境**
   ```bash
   pip install MetaTrader5 torch pandas numpy
   ```

### 使用步骤

#### 1. 首次运行（模拟账户）

```bash
# 1. 打开MT5并登录模拟账户
# 2. 运行交易程序
python live_trading.py

# 3. 确认启动
Type 'START' to begin live trading: START
```

#### 2. 程序运行中

程序会自动：
- 每小时检查一次市场（可配置）
- 生成交易信号
- 执行交易决策
- 显示账户状态

示例输出：
```
============================================================
Iteration #1
============================================================
2025-10-29 12:00:00 - INFO - Signal: BUY (Confidence: 65.32%)
2025-10-29 12:00:01 - INFO - Opening BUY position
2025-10-29 12:00:01 - INFO - ✓ Position opened: BUY
  Ticket: 12345678
  Price: 1.08500
  SL: 1.08000 (50 pips)
  TP: 1.09000 (100 pips)
  Volume: 0.01

------------------------------------------------------------
SYSTEM STATUS
------------------------------------------------------------
Time: 2025-10-29 12:00:02
Balance: $10000.00
Equity: $10005.50
Open Positions: 1
Daily Trades: 1/10

Current Positions:
  BUY | Ticket: 12345678 | Volume: 0.01 | Profit: $5.50
------------------------------------------------------------
```

#### 3. 停止程序

```
按 Ctrl+C 停止

Close all positions before exit? (y/n): y
所有持仓已平仓
系统已停止
```

---

## ⚙️ 配置参数

### 在 `live_trading.py` 中的 `TradingConfig` 类修改：

```python
class TradingConfig:
    # 交易品种
    SYMBOL = "EURUSD"
    
    # 交易参数
    LOT_SIZE = 0.01              # 手数（建议从0.01开始）
    STOP_LOSS_PIPS = 50          # 止损50点
    TAKE_PROFIT_PIPS = 100       # 止盈100点
    
    # 风险控制
    MAX_DAILY_TRADES = 10        # 每日最多10笔交易
    MAX_POSITIONS = 1            # 最多同时1个持仓
    TRADING_ENABLED = True       # 交易开关
    
    # 时间控制
    CHECK_INTERVAL = 3600        # 每小时检查一次
    TRADING_HOURS_START = 0      # 全天交易
    TRADING_HOURS_END = 24
```

### 推荐配置（新手）

```python
LOT_SIZE = 0.01                  # 最小手数
STOP_LOSS_PIPS = 30              # 较小止损
TAKE_PROFIT_PIPS = 60            # 风险收益比 1:2
MAX_DAILY_TRADES = 5             # 限制交易频率
MAX_POSITIONS = 1                # 一次只交易一个方向
CHECK_INTERVAL = 3600            # 每小时检查
```

### 推荐配置（保守型）

```python
LOT_SIZE = 0.05                  # 稍大手数
STOP_LOSS_PIPS = 50
TAKE_PROFIT_PIPS = 100
MAX_DAILY_TRADES = 10
MAX_POSITIONS = 2
TRADING_HOURS_START = 8          # 只在欧美盘交易
TRADING_HOURS_END = 22
```

---

## 📊 交易逻辑

### 信号类型

| 信号 | 说明 | 操作 |
|------|------|------|
| **BUY (1)** | 预测价格上涨 | 1. 平掉所有空头持仓<br>2. 开多头仓位（如果没有） |
| **SELL (0)** | 预测价格下跌 | 1. 平掉所有多头持仓<br>2. 开空头仓位（如果没有） |
| **HOLD (2)** | 预测价格横盘 | 平掉所有持仓 |

### 开仓条件

必须同时满足：
1. ✅ 信号置信度 > 40%
2. ✅ 在交易时间内
3. ✅ 未达到每日交易次数上限
4. ✅ 未达到最大持仓数量
5. ✅ 交易开关已启用

### 风险管理

每笔交易自动设置：
- **止损 (SL)**: 距离开仓价格 50 点
- **止盈 (TP)**: 距离开仓价格 100 点
- **风险收益比**: 1:2

---

## 📝 日志系统

### 日志文件

所有交易活动记录在 `trading_log.txt`：

```
2025-10-29 08:00:00 - INFO - System initialized successfully!
2025-10-29 09:00:00 - INFO - Signal: BUY (Confidence: 67.89%)
2025-10-29 09:00:01 - INFO - ✓ Position opened: BUY
2025-10-29 10:00:00 - INFO - Signal: HOLD (Confidence: 55.23%)
2025-10-29 10:00:01 - INFO - ✓ Position closed: Ticket=12345, Profit=$15.50
```

### 日志级别

- **INFO**: 正常操作信息
- **WARNING**: 警告信息（如达到交易限制）
- **ERROR**: 错误信息（如连接失败、下单失败）

---

## 🔧 故障排除

### 常见问题

**1. MT5初始化失败**
```
Error: MT5 initialization failed
```
**解决方案**:
- 确保MT5已打开并登录
- 检查是否使用管理员权限运行
- 重启MT5和程序

**2. 模型文件未找到**
```
Error: Model file not found: lstm_model.pth
```
**解决方案**:
- 先运行 `lstm_forex_backtest.py` 训练模型
- 确保 `lstm_model.pth` 和 `scaler.pkl` 在同一目录

**3. 下单失败**
```
Error: Order failed: Invalid stops
```
**解决方案**:
- 检查止损止盈距离是否符合券商要求
- 检查账户余额是否足够
- 检查交易品种是否允许交易

**4. 数据不足**
```
Warning: Insufficient data: got 45 bars
```
**解决方案**:
- 等待更多K线数据累积
- 程序需要至少60根H1 K线

### 紧急停止

**方法1**: 按 `Ctrl+C`
```
Stopping live trading...
Close all positions before exit? (y/n): y
```

**方法2**: 在代码中设置
```python
TRADING_ENABLED = False  # 禁用交易
```

---

## 📈 性能监控

### 关键指标

程序会实时显示：

1. **账户信息**
   - 余额 (Balance)
   - 净值 (Equity)
   - 已用保证金 (Margin)
   - 可用保证金 (Free Margin)

2. **交易统计**
   - 当前持仓数量
   - 今日交易次数
   - 浮动盈亏

3. **信号质量**
   - 信号类型 (BUY/SELL/HOLD)
   - 置信度百分比

### 建议监控

- **每日查看日志**: 检查交易记录
- **每周评估**: 统计胜率和盈亏
- **每月优化**: 根据表现调整参数

---

## 🛡️ 安全建议

### 必读安全规则

1. ✅ **永远从模拟账户开始**
2. ✅ **从最小手数开始（0.01）**
3. ✅ **设置每日交易次数限制**
4. ✅ **使用止损保护每笔交易**
5. ✅ **定期检查日志**
6. ✅ **不要依赖单一策略**
7. ✅ **保持充足的保证金**
8. ⚠️ **不要在新闻发布时交易**
9. ⚠️ **不要过度杠杆**
10. ⚠️ **始终保持理性**

### 资金管理

| 账户余额 | 建议手数 | 风险比例 |
|---------|---------|---------||
| $100 - $500 | 0.01 | 1-2% |
| $500 - $1000 | 0.01-0.02 | 1-2% |
| $1000 - $5000 | 0.02-0.05 | 1-2% |
| $5000+ | 0.05-0.10 | 1-2% |

---

## 🔄 更新和维护

### 定期任务

**每周**:
- 检查模型表现
- 清理旧日志文件
- 检查MT5连接稳定性

**每月**:
- 重新训练模型（使用最新数据）
- 更新 `lstm_model.pth` 和 `scaler.pkl`
- 评估策略参数

### 重新训练模型

```bash
# 1. 使用最新10年数据重新训练
python lstm_forex_backtest.py

# 2. 重启交易系统
python live_trading.py
```

---

## 📞 支持和反馈

### 系统要求

- Python 3.8+
- MetaTrader 5
- Windows 10/11 或 Linux (with Wine)
- 稳定的网络连接

### 文件清单

```
project/
├── live_trading.py           # 实盘交易主程序 ⭐
├── lstm_model.pth            # 训练好的模型（必需）
├── scaler.pkl                # 数据标准化器（必需）
├── trading_log.txt           # 交易日志（自动生成）
├── lstm_forex_backtest.py    # 模型训练脚本
└── LIVE_TRADING_README.md    # 本文件
```

---

## ⚖️ 免责声明

本软件仅供教育和研究目的使用。使用本软件进行实盘交易的所有风险由用户自行承担。

**重要提示**:
- 过去的表现不代表未来的结果
- 金融市场交易存在高风险
- 可能导致本金全部损失
- 作者不对任何直接或间接损失负责
- 使用前请咨询专业财务顾问

**使用本软件即表示您同意**:
- 您理解交易风险
- 您已在模拟账户充分测试
- 您对所有交易决策负完全责任
- 您不会对作者追究任何法律责任

---

## 🎓 学习资源

- [MetaTrader 5 文档](https://www.mql5.com/en/docs)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [外汇交易基础](https://www.investopedia.com/forex-trading-4689660)
- [风险管理策略](https://www.babypips.com/learn/forex/money-management)

---

**祝交易顺利！记住：风险管理永远是第一位的！** 🎯
