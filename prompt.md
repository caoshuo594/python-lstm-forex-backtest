# LSTM 外汇交易系统 - 开发文档

## 项目概述

这是一个基于深度学习的完整外汇交易系统，包含两个核心模块：

1. **训练与回测脚本** (`lstm_forex_backtest.py`): 数据获取 → 模型训练 → 高性能回测 → HTML报告
2. **实盘交易** (`live_trading.py`): MT5自动交易系统 ⚠️

---

## 系统架构

### 核心特性

**LSTM预测模型**:
- **输入**: 60根H1 K线的OHLC数据（序列长度=60）
- **架构**: 2层LSTM（隐藏层100个单元）+ 全连接层
- **输出**: 三分类信号 (BUY=1, SELL=0, HOLD=2)
- **训练数据**: 10年EURUSD历史数据

**高性能回测**:
- ✅ **信号预计算**: 回测前一次性批量预测所有信号，速度提升50-100倍
- ✅ **批量推理**: 使用1000个样本一批的批处理，优化内存和计算效率
- ✅ **避免重复计算**: 每个时间步只需查表，不再重复调用模型
- ✅ **交互式报告**: 自动生成HTML可视化报告（K线图、权益曲线、交易明细）

**风险管理**:
- 固定手数交易（0.01手）
- 交易佣金（0.02%）
- 初始资金（10万单位）

**[Technologies]**

*   `MetaTrader5` - 外汇数据获取和实盘交易接口
*   `pandas` - 数据处理和时间序列操作
*   `numpy` - 数值计算和数组操作
*   `torch` (PyTorch) - LSTM模型训练和推理
*   `sklearn` - 数据标准化（MinMaxScaler）
*   `backtesting` - 回测框架（内置bokeh用于HTML可视化）
*   `pickle` - 模型和scaler持久化

---

## 项目结构

生成完成后，项目应包含以下文件：

```
python-lstm-Backtesting/
├── lstm_forex_backtest.py      # 主训练和回测脚本
├── live_trading.py             # 实盘交易脚本
├── lstm_model.pth              # 训练好的LSTM模型（自动生成）
├── scaler.pkl                  # MinMaxScaler对象（自动生成）
├── backtest_report.html        # 交互式HTML报告（自动生成）
├── backtest_report.txt         # 文本格式报告（自动生成）
├── LIVE_TRADING_README.md      # 实盘交易使用说明
├── README.md                   # 项目说明文档
└── prompt.md                   # 本文件（开发规范）
```

---

## 工作流程规范

你必须严格遵循以下步骤和规范：

### 1. 数据获取 (MetaTrader 5)

*   导入 `MetaTrader5` 库并初始化连接 (`mt5.initialize()`)
*   使用 `datetime.datetime.now()` 和 `datetime.timedelta(days=365*10)` 计算10年的时间范围
*   从MT5获取 `EURUSD` 的 `TIMEFRAME_H1` 数据，使用 `mt5.copy_rates_range()`
*   将数据转换为 `pandas.DataFrame`

**关键数据帧规范：**
*   必须将 `time` 列设为 `pandas.datetime` 索引 (使用 `pd.to_datetime(..., unit='s')`)
*   必须将列重命名为 `backtesting.py` 要求的格式：`Open`, `High`, `Low`, `Close` (首字母大写)
*   只保留 'Open', 'High', 'Low', 'Close', 'tick_volume' 列，将 'tick_volume' 重命名为 `Volume`

### 2. 数据预处理与特征工程

*   **特征定义**: 使用 `Open`, `High`, `Low`, `Close` 四个特征作为模型输入
*   **目标变量 (Label) 定义** - 三分类问题：

    *   未来查找周期: `TARGET_PERIODS = 6` (6小时后)
    *   价格变动阈值: `THRESHOLD = 0.0005` (0.05%)
    *   **Labeling 逻辑**:
        ```python
        price_change = (close[i+6] - close[i]) / close[i]
        if price_change > 0.0005:    label = 1  # Buy
        elif price_change < -0.0005: label = 0  # Sell
        else:                        label = 2  # Hold
        ```

*   **数据标准化**:
    *   使用 `sklearn.preprocessing.MinMaxScaler`
    *   仅对特征 (`Open`, `High`, `Low`, `Close`) 进行 `fit_transform`
    *   **必须保存** scaler对象到 `scaler.pkl`，供回测和实盘使用

*   **序列创建**:
    *   序列长度: `SEQUENCE_LENGTH = 60` (模型回看60个H1 K线)
    *   创建函数 `create_sequences(features, labels, seq_length)`
    *   输出形状: X=(N, 60, 4), y=(N,)

*   **数据拆分**:
    *   **严格按时间顺序**拆分，禁止随机打乱
    *   训练集/测试集比例: `SPLIT_RATIO = 0.8` (前80%训练，后20%测试)

*   **PyTorch Dataset 和 DataLoader**:
    *   将数据转换为 `torch.FloatTensor` (X) 和 `torch.LongTensor` (y)
    *   创建自定义 `TimeSeriesDataset` 类
    *   创建 `DataLoader`，批大小 `BATCH_SIZE = 64`

### 3. LSTM模型定义

**架构规范**:

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out
```

**参数配置**:
*   输入维度: 4 (OHLC)
*   隐藏层大小: 100
*   LSTM层数: 2
*   Dropout: 0.2
*   输出类别数: 3

### 4. 模型训练

**训练配置**:
*   损失函数: `nn.CrossEntropyLoss()`
*   优化器: `torch.optim.Adam`, 学习率 `LEARNING_RATE = 0.001`
*   训练轮数: `NUM_EPOCHS = 20`
*   设备: 自动检测 CUDA（如可用）或 CPU

**训练流程**:
1. 每个epoch进行训练和测试
2. 记录训练损失和测试准确率
3. **保存最佳模型**: 当测试准确率提升时，保存模型到 `lstm_model.pth`
4. 打印每个epoch的统计信息

**输出示例**:
```
Epoch [1/20], Loss: 0.8234, Test Accuracy: 52.31%
Epoch [2/20], Loss: 0.7156, Test Accuracy: 56.78%
...
Best test accuracy: 58.45%
```

### 5. 高性能回测系统

**关键优化 - 信号预计算**:

传统方法的问题：
- 在 `Strategy.next()` 中每个bar都调用模型
- 大量重复的数据转换和模型推理
- 10年数据约87,600次模型调用，非常慢

**优化方案**:

```python
def precompute_signals(df, model_path='lstm_model.pth', scaler_path='scaler.pkl'):
    """预先计算所有交易信号，显著提升回测速度"""
    
    # 1. 加载模型和scaler
    model = LSTMModel(...)
    model.load_state_dict(torch.load(model_path, ...))
    model.eval()
    
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # 2. 一次性缩放所有特征数据
    features = df[['Open', 'High', 'Low', 'Close']].values
    scaled_features = scaler.transform(features)
    
    # 3. 收集所有有效序列
    batch_sequences = []
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        sequence = scaled_features[i-SEQUENCE_LENGTH:i]
        batch_sequences.append(sequence)
    
    # 4. 批量预测（1000个样本一批）
    tensor_data = torch.FloatTensor(np.array(batch_sequences))
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_data), 1000):
            batch = tensor_data[i:i+1000]
            outputs = model(batch)
            batch_predictions = torch.argmax(outputs, dim=1).numpy()
            predictions.extend(batch_predictions)
    
    # 5. 返回信号数组
    return signals  # numpy array of shape (len(df),)
```

**回测策略**:

```python
class LstmStrategy(Strategy):
    signals = None  # 类变量，存储预计算的信号
    
    def init(self):
        # 验证信号已预计算
        if LstmStrategy.signals is None:
            raise ValueError("Signals not precomputed!")
        
        # 将信号转换为indicator（可在图表上显示）
        self.signal = self.I(lambda: LstmStrategy.signals)
    
    def next(self):
        current_idx = len(self.data.Close) - 1
        
        # 直接查表获取信号（无需模型推理）
        signal = LstmStrategy.signals[current_idx]
        
        # 执行交易逻辑
        if signal == 1:  # Buy
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy()
        elif signal == 0:  # Sell
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell()
        elif signal == 2:  # Hold
            if self.position:
                self.position.close()
```

**性能提升**:
- 原方法: 每个bar都调用模型（~87,600次） → 可能需要30-60分钟
- 优化后: 预先批量计算 + 查表 → **1-2分钟完成**
- **速度提升**: 50-100倍

### 6. 回测执行

**配置参数**:
```python
bt = Backtest(
    df,                    # 数据框
    LstmStrategy,          # 策略类
    cash=100000,          # 初始资金
    commission=0.0002,    # 手续费（0.02%）
    exclusive_orders=True # 同时只能有一个方向的订单
)
```

**执行流程**:
1. 预计算所有信号: `signals = precompute_signals(df, ...)`
2. 将信号存储到策略类: `LstmStrategy.signals = signals`
3. 运行回测: `stats = bt.run()`
4. 生成报告: 
   - 文本报告保存到 `backtest_report.txt`
   - HTML报告保存到 `backtest_report.html`

### 7. 报告生成

**文本报告** (backtest_report.txt):
- 总收益率
- 夏普比率
- 最大回撤
- 交易次数
- 胜率等统计指标

**HTML报告** (backtest_report.html):
- 交互式K线图（可缩放、平移）
- 买入/卖出信号标记
- 权益曲线
- 回撤曲线
- 交易明细表

**生成代码**:
```python
try:
    bt.plot(filename='backtest_report.html', open_browser=False, resample=False)
    print("[SUCCESS] Interactive HTML report saved")
except Exception as e:
    print(f"[WARNING] Plot generation issue: {e}")
    # 尝试简化版本
    bt.plot(filename='backtest_report.html', open_browser=False, 
            plot_width=None, plot_equity=True)
```

---

## 实盘交易规范

### live_trading.py

**核心功能**:
- 从MT5实时获取H1数据
- 使用训练好的LSTM模型生成信号
- 自动下单、管理止损止盈
- 每小时检查一次新信号

**风险管理**:
- 固定手数: 0.01手
- 止损: 50 pips
- 止盈: 100 pips (2:1 盈亏比)
- 信号置信度阈值: 40%
- 最大持仓限制: 1个

**使用前必读**:
1. ⚠️ **先在模拟账户测试**
2. ⚠️ **理解所有风险**
3. ⚠️ **监控系统运行**
4. 详细说明见 `LIVE_TRADING_README.md`

---

## 常量定义

```python
# LSTM模型参数
SEQUENCE_LENGTH = 60      # 序列长度（60根H1 K线）
HIDDEN_SIZE = 100         # LSTM隐藏层大小
NUM_LAYERS = 2            # LSTM层数
BATCH_SIZE = 64           # 训练批大小
NUM_EPOCHS = 20           # 训练轮数
LEARNING_RATE = 0.001     # 学习率

# 标签生成参数
TARGET_PERIODS = 6        # 未来6小时
THRESHOLD = 0.0005        # 0.05% 价格变动阈值

# 数据拆分
SPLIT_RATIO = 0.8         # 80% 训练，20% 测试

# 回测参数
INITIAL_CASH = 100000     # 初始资金
COMMISSION = 0.0002       # 0.02% 手续费
```

---

## 输出示例

```
Step 1: Fetching data from MT5...
Data shape: (87649, 5)

Step 2: Creating labels...

Step 3: Preprocessing data...

Step 4: Creating sequences...
Sequences shape: X=(87589, 60, 4), y=(87589,)
Train: (70071, 60, 4), Test: (17518, 60, 4)

Step 5: Training LSTM model...
Using device: cuda
Epoch [1/20], Loss: 0.8234, Test Accuracy: 52.31%
Epoch [2/20], Loss: 0.7156, Test Accuracy: 56.78%
...
Best test accuracy: 58.45%

Step 6: Precomputing trading signals for backtest...
✓ Precomputed 87589 signals
  - Buy signals: 15234
  - Sell signals: 14567
  - Hold signals: 57788

Step 7: Running backtest...
==================================================
BACKTEST RESULTS
==================================================
Return [%]                    15.34
Sharpe Ratio                   1.23
Max Drawdown [%]              12.45
# Trades                        342
Win Rate [%]                  54.39
...

[SUCCESS] Interactive HTML report saved to: backtest_report.html
```

---

## 性能优化要点

### 1. 回测性能优化
- ✅ 使用信号预计算代替逐步推理
- ✅ 批量处理（1000个样本/批）
- ✅ 避免重复的数据转换
- ✅ 使用numpy数组存储信号
- 🚫 不要在 `Strategy.next()` 中调用模型

### 2. 模型训练优化
- ✅ 使用GPU加速（如可用）
- ✅ 合理的批大小（64）
- ✅ 保存最佳模型，避免过拟合
- ✅ 早停机制（如需要）

### 3. 数据处理优化
- ✅ 一次性标准化所有特征
- ✅ 使用numpy而非pandas进行数值计算
- ✅ 避免循环中的重复操作

---

## 注意事项

### 开发规范
1. 所有numpy随机操作都设置seed以确保可复现性
2. 时间序列数据严禁打乱顺序
3. 必须保存scaler对象供回测和实盘使用
4. 回测前必须预计算信号

### 文件管理
1. `lstm_model.pth` - 模型权重（约400KB）
2. `scaler.pkl` - 数据标准化器（约1KB）
3. `backtest_report.html` - HTML报告（约1-2MB）
4. `backtest_report.txt` - 文本报告（约2KB）

### 实盘交易警告
- ⚠️ 使用真实资金前必须充分测试
- ⚠️ 市场条件可能与历史数据不同
- ⚠️ 监控系统运行状态
- ⚠️ 准备好手动干预计划

---

## 技术支持

如遇问题，检查：
1. MT5是否正确安装和登录
2. Python库是否完整安装
3. 模型文件是否存在
4. 数据获取是否成功

---

## 免责声明

本系统仅供学习和研究使用。外汇交易存在高风险，可能导致全部本金损失。使用本系统进行实盘交易的所有风险和责任由用户自行承担。

---

**版本**: v2.0  
**更新日期**: 2025-01-03  
**优化**: 高性能回测系统，速度提升50-100倍
