import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import time
import torch
import torch.nn as nn
import pickle
import logging
from pathlib import Path

"""
LSTM Forex Live Trading System
实时交易系统 - 连接MT5进行自动交易

警告: 这是实盘交易程序，会进行真实交易！
建议先在模拟账户测试！
"""

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()

# ==================== 配置参数 ====================

class TradingConfig:
    """交易配置参数"""
    
    # MT5配置
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_H1
    MAGIC_NUMBER = 234000  # 唯一标识符，用于识别此EA的订单
    
    # 模型配置
    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    
    # 交易配置
    LOT_SIZE = 0.01  # 交易手数（0.01 = 微型手）
    STOP_LOSS_PIPS = 50  # 止损点数
    TAKE_PROFIT_PIPS = 100  # 止盈点数
    MAX_SLIPPAGE = 10  # 最大滑点
    
    # 风险管理
    MAX_DAILY_TRADES = 10  # 每日最大交易次数
    MAX_POSITIONS = 1  # 最大持仓数量
    TRADING_ENABLED = True  # 交易开关
    
    # 时间控制
    CHECK_INTERVAL = 3600  # 检查间隔（秒），3600 = 1小时
    TRADING_HOURS_START = 0  # 交易时间开始（小时）
    TRADING_HOURS_END = 24  # 交易时间结束（小时）
    
    # 文件路径（使用绝对路径）
    MODEL_PATH = str(SCRIPT_DIR / "lstm_model.pth")
    SCALER_PATH = str(SCRIPT_DIR / "scaler.pkl")
    LOG_PATH = str(SCRIPT_DIR / "trading_log.txt")


# ==================== LSTM模型定义 ====================

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


# ==================== 交易系统类 ====================

class LiveTradingSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_PATH, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """初始化系统"""
        self.logger.info("="*60)
        self.logger.info("LSTM Live Trading System - Initializing")
        self.logger.info("="*60)
        
        # 检查模型文件
        if not Path(self.config.MODEL_PATH).exists():
            self.logger.error(f"Model file not found: {self.config.MODEL_PATH}")
            return False
        
        if not Path(self.config.SCALER_PATH).exists():
            self.logger.error(f"Scaler file not found: {self.config.SCALER_PATH}")
            return False
        
        # 加载模型
        try:
            self.model = LSTMModel(
                input_size=4,
                hidden_size=self.config.HIDDEN_SIZE,
                num_layers=self.config.NUM_LAYERS,
                num_classes=3
            )
            self.model.load_state_dict(torch.load(
                self.config.MODEL_PATH,
                map_location='cpu',
                weights_only=False
            ))
            self.model.eval()
            self.logger.info("[OK] Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
        
        # 加载scaler
        try:
            with open(self.config.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            self.logger.info("[OK] Scaler loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")
            return False
        
        # 初始化MT5
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        
        # 获取账户信息
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return False
        
        self.logger.info(f"[OK] MT5 connected")
        self.logger.info(f"Account: {account_info.login}")
        self.logger.info(f"Balance: ${account_info.balance:.2f}")
        self.logger.info(f"Equity: ${account_info.equity:.2f}")
        self.logger.info(f"Server: {account_info.server}")
        
        # 检查交易品种
        symbol_info = mt5.symbol_info(self.config.SYMBOL)
        if symbol_info is None:
            self.logger.error(f"Symbol {self.config.SYMBOL} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.config.SYMBOL, True):
                self.logger.error(f"Failed to select symbol {self.config.SYMBOL}")
                return False
        
        self.logger.info(f"[OK] Symbol {self.config.SYMBOL} ready")
        self.logger.info(f"Spread: {symbol_info.spread} points")
        
        self.logger.info("="*60)
        self.logger.info("System initialized successfully!")
        self.logger.info("="*60)
        
        return True
    
    def get_historical_data(self):
        """获取历史数据用于预测"""
        try:
            # 获取足够的历史数据
            rates = mt5.copy_rates_from_pos(
                self.config.SYMBOL,
                self.config.TIMEFRAME,
                0,  # 从最新K线开始
                self.config.SEQUENCE_LENGTH + 10  # 多获取一些以防万一
            )
            
            if rates is None or len(rates) < self.config.SEQUENCE_LENGTH:
                self.logger.warning(f"Insufficient data: got {len(rates) if rates is not None else 0} bars")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    def generate_signal(self):
        """生成交易信号"""
        try:
            # 获取数据
            df = self.get_historical_data()
            if df is None:
                return None
            
            # 准备特征
            features = np.column_stack([
                df['open'].values[-self.config.SEQUENCE_LENGTH:],
                df['high'].values[-self.config.SEQUENCE_LENGTH:],
                df['low'].values[-self.config.SEQUENCE_LENGTH:],
                df['close'].values[-self.config.SEQUENCE_LENGTH:]
            ])
            
            # 标准化
            scaled_features = self.scaler.transform(features)
            
            # 转换为tensor
            tensor_data = torch.FloatTensor(scaled_features).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(tensor_data)
                probabilities = torch.softmax(outputs, dim=1)
                signal = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][signal].item()
            
            signal_names = {0: "SELL", 1: "BUY", 2: "HOLD"}
            
            self.logger.info(f"Signal: {signal_names[signal]} (Confidence: {confidence:.2%})")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'signal_name': signal_names[signal],
                'timestamp': datetime.datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def get_current_positions(self):
        """获取当前持仓"""
        positions = mt5.positions_get(symbol=self.config.SYMBOL)
        if positions is None:
            return []
        return [pos for pos in positions if pos.magic == self.config.MAGIC_NUMBER]
    
    def close_position(self, position):
        """平仓"""
        try:
            # 准备平仓请求
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            price = mt5.symbol_info_tick(self.config.SYMBOL).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.config.SYMBOL).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": self.config.MAX_SLIPPAGE,
                "magic": self.config.MAGIC_NUMBER,
                "comment": "Close by LSTM",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close failed: {result.comment}")
                return False
            
            profit = position.profit
            self.logger.info(f"[OK] Position closed: Ticket={position.ticket}, Profit=${profit:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def open_position(self, signal_type):
        """开仓"""
        try:
            # 检查是否允许交易
            if not self.config.TRADING_ENABLED:
                self.logger.info("Trading is disabled")
                return False
            
            # 检查每日交易次数
            today = datetime.date.today()
            if self.last_trade_date != today:
                self.daily_trades_count = 0
                self.last_trade_date = today
            
            if self.daily_trades_count >= self.config.MAX_DAILY_TRADES:
                self.logger.warning(f"Daily trade limit reached: {self.config.MAX_DAILY_TRADES}")
                return False
            
            # 检查持仓数量
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.config.MAX_POSITIONS:
                self.logger.warning(f"Max positions reached: {self.config.MAX_POSITIONS}")
                return False
            
            # 获取当前价格
            tick = mt5.symbol_info_tick(self.config.SYMBOL)
            if tick is None:
                self.logger.error("Failed to get current price")
                return False
            
            symbol_info = mt5.symbol_info(self.config.SYMBOL)
            point = symbol_info.point
            
            # 确定订单类型和价格
            if signal_type == 1:  # BUY
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - self.config.STOP_LOSS_PIPS * point * 10
                tp = price + self.config.TAKE_PROFIT_PIPS * point * 10
            elif signal_type == 0:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + self.config.STOP_LOSS_PIPS * point * 10
                tp = price - self.config.TAKE_PROFIT_PIPS * point * 10
            else:
                return False
            
            # 准备订单请求
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.SYMBOL,
                "volume": self.config.LOT_SIZE,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.config.MAX_SLIPPAGE,
                "magic": self.config.MAGIC_NUMBER,
                "comment": "LSTM Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 发送订单
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return False
            
            self.daily_trades_count += 1
            
            order_name = "BUY" if signal_type == 1 else "SELL"
            self.logger.info(f"[OK] Position opened: {order_name}")
            self.logger.info(f"  Ticket: {result.order}")
            self.logger.info(f"  Price: {price:.5f}")
            self.logger.info(f"  SL: {sl:.5f} ({self.config.STOP_LOSS_PIPS} pips)")
            self.logger.info(f"  TP: {tp:.5f} ({self.config.TAKE_PROFIT_PIPS} pips)")
            self.logger.info(f"  Volume: {self.config.LOT_SIZE}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return False
    
    def is_trading_hours(self):
        """检查是否在交易时间内"""
        current_hour = datetime.datetime.now().hour
        return self.config.TRADING_HOURS_START <= current_hour < self.config.TRADING_HOURS_END
    
    def execute_trading_logic(self):
        """执行交易逻辑"""
        try:
            # 检查交易时间
            if not self.is_trading_hours():
                self.logger.info("Outside trading hours")
                return
            
            # 生成信号
            signal_data = self.generate_signal()
            if signal_data is None:
                return
            
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # 获取当前持仓
            current_positions = self.get_current_positions()
            
            # 交易逻辑
            if signal == 1:  # BUY信号
                # 如果有空头持仓，先平仓
                for pos in current_positions:
                    if pos.type == mt5.ORDER_TYPE_SELL:
                        self.logger.info("Closing SELL position due to BUY signal")
                        self.close_position(pos)
                
                # 如果没有多头持仓，开仓
                has_long = any(pos.type == mt5.ORDER_TYPE_BUY for pos in current_positions)
                if not has_long and confidence > 0.4:  # 信心阈值
                    self.logger.info("Opening BUY position")
                    self.open_position(1)
            
            elif signal == 0:  # SELL信号
                # 如果有多头持仓，先平仓
                for pos in current_positions:
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        self.logger.info("Closing BUY position due to SELL signal")
                        self.close_position(pos)
                
                # 如果没有空头持仓，开仓
                has_short = any(pos.type == mt5.ORDER_TYPE_SELL for pos in current_positions)
                if not has_short and confidence > 0.4:  # 信心阈值
                    self.logger.info("Opening SELL position")
                    self.open_position(0)
            
            elif signal == 2:  # HOLD信号
                # 平掉所有持仓
                if current_positions:
                    self.logger.info("HOLD signal - closing all positions")
                    for pos in current_positions:
                        self.close_position(pos)
            
        except Exception as e:
            self.logger.error(f"Error in trading logic: {e}")
    
    def print_status(self):
        """打印当前状态"""
        try:
            account_info = mt5.account_info()
            positions = self.get_current_positions()
            
            self.logger.info("-"*60)
            self.logger.info("SYSTEM STATUS")
            self.logger.info("-"*60)
            self.logger.info(f"Time: {datetime.datetime.now()}")
            self.logger.info(f"Balance: ${account_info.balance:.2f}")
            self.logger.info(f"Equity: ${account_info.equity:.2f}")
            self.logger.info(f"Margin: ${account_info.margin:.2f}")
            self.logger.info(f"Free Margin: ${account_info.margin_free:.2f}")
            self.logger.info(f"Open Positions: {len(positions)}")
            self.logger.info(f"Daily Trades: {self.daily_trades_count}/{self.config.MAX_DAILY_TRADES}")
            
            if positions:
                self.logger.info("\nCurrent Positions:")
                for pos in positions:
                    pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    self.logger.info(f"  {pos_type} | Ticket: {pos.ticket} | "
                                   f"Volume: {pos.volume} | Profit: ${pos.profit:.2f}")
            
            self.logger.info("-"*60)
            
        except Exception as e:
            self.logger.error(f"Error printing status: {e}")
    
    def run(self):
        """运行交易系统"""
        if not self.initialize():
            self.logger.error("Initialization failed. Exiting.")
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("LIVE TRADING STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Symbol: {self.config.SYMBOL}")
        self.logger.info(f"Timeframe: H1")
        self.logger.info(f"Check Interval: {self.config.CHECK_INTERVAL}s")
        self.logger.info(f"Lot Size: {self.config.LOT_SIZE}")
        self.logger.info(f"Stop Loss: {self.config.STOP_LOSS_PIPS} pips")
        self.logger.info(f"Take Profit: {self.config.TAKE_PROFIT_PIPS} pips")
        self.logger.info("="*60)
        self.logger.info("\nPress Ctrl+C to stop\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Iteration #{iteration}")
                self.logger.info(f"{'='*60}")
                
                # 执行交易逻辑
                self.execute_trading_logic()
                
                # 打印状态
                self.print_status()
                
                # 等待下一次检查
                self.logger.info(f"\nWaiting {self.config.CHECK_INTERVAL}s for next check...")
                time.sleep(self.config.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("\n" + "="*60)
            self.logger.info("STOPPING LIVE TRADING")
            self.logger.info("="*60)
            
            # 可选：退出时平掉所有持仓
            close_on_exit = input("\nClose all positions before exit? (y/n): ")
            if close_on_exit.lower() == 'y':
                positions = self.get_current_positions()
                for pos in positions:
                    self.close_position(pos)
                self.logger.info("All positions closed")
            
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
            self.logger.info("System stopped")
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            mt5.shutdown()


# ==================== 主程序 ====================

def main():
    """主函数"""
    print("="*60)
    print("LSTM FOREX LIVE TRADING SYSTEM")
    print("="*60)
    print("\n[WARNING] This will perform REAL TRADES!")
    print("[WARNING] Make sure you are using a DEMO account first!")
    print("\nRecommended settings for demo testing:")
    print("  - LOT_SIZE = 0.01 (micro lot)")
    print("  - MAX_POSITIONS = 1")
    print("  - MAX_DAILY_TRADES = 10")
    print("\n" + "="*60)
    
    # 确认
    confirm = input("\nType 'START' to begin live trading: ")
    if confirm.upper() != 'START':
        print("Cancelled by user")
        return
    
    # 创建配置
    config = TradingConfig()
    
    # 创建交易系统
    trading_system = LiveTradingSystem(config)
    
    # 运行
    trading_system.run()


if __name__ == "__main__":
    main()
