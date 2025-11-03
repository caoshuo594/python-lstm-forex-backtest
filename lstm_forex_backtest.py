import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from backtesting import Backtest, Strategy
import pickle


# Constants
TARGET_PERIODS = 6
THRESHOLD = 0.0005
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
HIDDEN_SIZE = 100
NUM_LAYERS = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
SPLIT_RATIO = 0.8


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(features, labels, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)


def get_mt5_data():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None
    
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * 10)
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("Failed to get data from MT5")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df


def create_labels(df):
    close_prices = df['Close'].values
    labels = np.full(len(close_prices), 2)
    
    for i in range(len(close_prices) - TARGET_PERIODS):
        price_change = (close_prices[i + TARGET_PERIODS] - close_prices[i]) / close_prices[i]
        if price_change > THRESHOLD:
            labels[i] = 1
        elif price_change < -THRESHOLD:
            labels[i] = 0
    
    return labels


def train_model(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMModel(input_size=4, hidden_size=HIDDEN_SIZE, 
                     num_layers=NUM_LAYERS, num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'lstm_model.pth')
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return model


class LstmStrategy(Strategy):
    signals = None  # 类变量，存储预先计算的信号
    
    def init(self):
        # 使用预先计算的信号（从类变量获取）
        if LstmStrategy.signals is None:
            raise ValueError("Signals not precomputed. Call precompute_signals() first.")
        
        # 将信号转换为indicator，这样可以在图表上显示
        self.signal = self.I(lambda: LstmStrategy.signals)
    
    def next(self):
        current_idx = len(self.data.Close) - 1
        
        # 检查索引是否有效
        if current_idx >= len(LstmStrategy.signals):
            return
        
        signal = LstmStrategy.signals[current_idx]
        
        if signal == 1:  # 买入信号
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy()
        elif signal == 0:  # 卖出信号
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell()
        elif signal == 2:  # 持有/平仓信号
            if self.position:
                self.position.close()


def precompute_signals(df, model_path='lstm_model.pth', scaler_path='scaler.pkl'):
    """预先计算所有交易信号，显著提升回测速度"""
    print("Precomputing all trading signals...")
    
    # 加载模型
    model = LSTMModel(input_size=4, hidden_size=HIDDEN_SIZE, 
                     num_layers=NUM_LAYERS, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    
    # 加载scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 准备特征数据
    features = df[['Open', 'High', 'Low', 'Close']].values
    scaled_features = scaler.transform(features)
    
    # 初始化信号数组（默认为持有）
    signals = np.full(len(df), 2, dtype=np.int32)
    
    # 批量预测
    device = torch.device("cpu")  # 回测用CPU即可
    batch_sequences = []
    valid_indices = []
    
    # 收集所有有效的序列
    for i in range(SEQUENCE_LENGTH, len(scaled_features)):
        sequence = scaled_features[i-SEQUENCE_LENGTH:i]
        batch_sequences.append(sequence)
        valid_indices.append(i)
    
    # 转换为tensor并批量预测
    if len(batch_sequences) > 0:
        tensor_data = torch.FloatTensor(np.array(batch_sequences)).to(device)
        
        with torch.no_grad():
            # 分批处理，避免内存溢出
            batch_size = 1000
            predictions = []
            
            for i in range(0, len(tensor_data), batch_size):
                batch = tensor_data[i:i+batch_size]
                outputs = model(batch)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        # 将预测结果填入信号数组
        for idx, pred in zip(valid_indices, predictions):
            signals[idx] = pred
    
    print(f"✓ Precomputed {len(valid_indices)} signals")
    print(f"  - Buy signals: {np.sum(signals == 1)}")
    print(f"  - Sell signals: {np.sum(signals == 0)}")
    print(f"  - Hold signals: {np.sum(signals == 2)}")
    
    return signals


def main():
    print("Step 1: Fetching data from MT5...")
    df = get_mt5_data()
    if df is None:
        print("Failed to get data. Exiting.")
        return
    print(f"Data shape: {df.shape}")
    
    print("\nStep 2: Creating labels...")
    labels = create_labels(df)
    
    print("\nStep 3: Preprocessing data...")
    features_df = df[['Open', 'High', 'Low', 'Close']].copy()
    
    valid_length = len(labels) - TARGET_PERIODS
    features_df = features_df.iloc[:valid_length]
    labels = labels[:valid_length]
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nStep 4: Creating sequences...")
    X, y = create_sequences(features_scaled, labels, SEQUENCE_LENGTH)
    print(f"Sequences shape: X={X.shape}, y={y.shape}")
    
    split_idx = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nStep 5: Training LSTM model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    print("\nStep 6: Precomputing trading signals for backtest...")
    signals = precompute_signals(df, 'lstm_model.pth', 'scaler.pkl')
    LstmStrategy.signals = signals  # 将信号存储到策略类
    
    print("\nStep 7: Running backtest...")
    bt = Backtest(df, LstmStrategy, cash=100000, commission=0.0002, 
                  exclusive_orders=True)
    stats = bt.run()
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(stats)
    
    with open('backtest_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("BACKTEST RESULTS\n")
        f.write("="*50 + "\n")
        f.write(str(stats) + "\n")
    print("\nBacktest report saved to: backtest_report.txt")
    
    print("\nGenerating interactive HTML report...")
    try:
        bt.plot(filename='backtest_report.html', open_browser=False, resample=False)
        print("[SUCCESS] Interactive HTML report saved to: backtest_report.html")
        print("          Open this file in your browser to view the interactive charts!")
    except Exception as e:
        print(f"[WARNING] Plot generation encountered an issue: {str(e)[:100]}")
        try:
            bt.plot(filename='backtest_report.html', open_browser=False, plot_width=None, plot_equity=True, plot_return=False, plot_pl=True)
            print("[SUCCESS] Simplified HTML report saved to: backtest_report.html")
        except:
            print("          Text report has been saved to backtest_report.txt")


if __name__ == "__main__":
    main()
