"""
Signal Notification System for A-Share Market
Manages and delivers buy/sell signals to users
"""
import smtplib
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import threading
import time
from typing import Dict, List, Optional
import pandas as pd


class SignalNotifier:
    """
    Signal notification system that tracks and sends alerts for buy/sell signals
    """
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self.setup_database()
        self.active_alerts = []
    
    def setup_database(self):
        """
        Setup SQLite database to store signal notifications
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing signals
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT,
                signal_type TEXT NOT NULL,  -- BUY, SELL, HOLD
                strategy TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                price REAL,
                reason TEXT,
                priority INTEGER DEFAULT 1  -- 1: Low, 2: Medium, 3: High
            )
        ''')
        
        # Create table for alert subscriptions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                phone TEXT,
                telegram_id TEXT,
                active BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_signal(self, symbol: str, name: str, signal_type: str, strategy: str, 
                   price: float = None, reason: str = "", priority: int = 1):
        """
        Add a new signal to the database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (symbol, name, signal_type, strategy, price, reason, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, name, signal_type, strategy, price, reason, priority))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"🔔 新信号记录: {symbol} - {signal_type} ({strategy})")
        
        # Trigger notification
        self._trigger_notification(signal_id, symbol, signal_type, strategy, price, reason)
        
        return signal_id
    
    def _trigger_notification(self, signal_id: int, symbol: str, signal_type: str, 
                             strategy: str, price: float = None, reason: str = ""):
        """
        Trigger notification for a new signal
        """
        # Get subscriber list
        subscribers = self.get_subscribers()
        
        subject = f"A股信号提醒: {symbol} - {signal_type}"
        body = self._create_notification_message(symbol, signal_type, strategy, price, reason)
        
        # Send notifications to all subscribers
        for subscriber in subscribers:
            if subscriber['email']:
                self._send_email_notification(subscriber['email'], subject, body)
            if subscriber['phone']:
                self._send_sms_notification(subscriber['phone'], body)
            if subscriber['telegram_id']:
                self._send_telegram_notification(subscriber['telegram_id'], body)
    
    def _create_notification_message(self, symbol: str, signal_type: str, strategy: str, 
                                    price: float = None, reason: str = ""):
        """
        Create notification message
        """
        signal_emoji = "📈" if signal_type.upper() == "BUY" else "📉" if signal_type.upper() == "SELL" else "⏸️"
        
        message = f"""
{signal_emoji} A股交易信号提醒

股票: {symbol}
信号类型: {signal_type}
策略: {strategy}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if price:
            message += f"当前价格: ¥{price:.2f}\n"
        
        if reason:
            message += f"原因: {reason}\n"
        
        message += "\n请根据您的风险偏好和投资策略进行决策。"
        
        return message
    
    def _send_email_notification(self, email: str, subject: str, body: str):
        """
        Send email notification
        """
        # This is a simplified email sending mechanism
        # In production, you would use proper SMTP settings
        print(f"📧 邮件发送至 {email}: {subject}")
        # Actual implementation would require SMTP configuration
    
    def _send_sms_notification(self, phone: str, message: str):
        """
        Send SMS notification (placeholder)
        """
        print(f"📱 短信发送至 {phone}: {message[:50]}...")
        # Actual implementation would require SMS gateway
    
    def _send_telegram_notification(self, telegram_id: str, message: str):
        """
        Send Telegram notification (placeholder)
        """
        print(f"💬 电报发送至 {telegram_id}: {message[:50]}...")
        # Actual implementation would require Telegram Bot API
    
    def get_subscribers(self) -> List[Dict]:
        """
        Get list of subscribers
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM subscriptions WHERE active = 1")
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        subscribers = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return subscribers
    
    def add_subscriber(self, email: str = None, phone: str = None, telegram_id: str = None):
        """
        Add a new subscriber
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO subscriptions (email, phone, telegram_id, active)
                VALUES (?, ?, ?, 1)
            ''', (email, phone, telegram_id))
            
            conn.commit()
            print(f"✅ 新订阅者添加: email={email}, phone={phone}, telegram={telegram_id}")
        except Exception as e:
            print(f"❌ 添加订阅者失败: {e}")
        finally:
            conn.close()
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """
        Get recent signals
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        signals = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return signals
    
    def get_signals_by_stock(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get signals for a specific stock
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals 
            WHERE symbol = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (symbol, limit))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        signals = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return signals
    
    def monitor_signals_continuously(self, check_interval: int = 300):
        """
        Continuously monitor for new signals (runs in background thread)
        """
        def monitor():
            while True:
                # In a real implementation, this would periodically check for new signals
                # from various strategies and market conditions
                time.sleep(check_interval)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        print(f"🔄 信号监控启动，检查间隔: {check_interval}秒")


class SignalProcessor:
    """
    Processes signals from various strategies and generates notifications
    """
    def __init__(self, notifier: SignalNotifier):
        self.notifier = notifier
    
    def process_strategy_signals(self, symbol: str, name: str, strategy_name: str, 
                               data: pd.DataFrame, signals: pd.Series):
        """
        Process signals from a strategy and generate notifications
        """
        if signals.empty:
            return
        
        # Process each signal
        for date, signal_value in signals.items():
            if signal_value == 1:  # Buy signal
                price = data.loc[date, 'close'] if 'close' in data.columns else None
                reason = f"策略 {strategy_name} 产生买入信号"
                self.notifier.add_signal(
                    symbol=symbol,
                    name=name,
                    signal_type="BUY",
                    strategy=strategy_name,
                    price=price,
                    reason=reason,
                    priority=2
                )
            elif signal_value == -1:  # Sell signal
                price = data.loc[date, 'close'] if 'close' in data.columns else None
                reason = f"策略 {strategy_name} 产生卖出信号"
                self.notifier.add_signal(
                    symbol=symbol,
                    name=name,
                    signal_type="SELL",
                    strategy=strategy_name,
                    price=price,
                    reason=reason,
                    priority=2
                )
    
    def generate_custom_alert(self, symbol: str, name: str, signal_type: str, 
                            strategy: str, price: float = None, reason: str = "", 
                            priority: int = 1):
        """
        Generate a custom alert
        """
        self.notifier.add_signal(
            symbol=symbol,
            name=name,
            signal_type=signal_type,
            strategy=strategy,
            price=price,
            reason=reason,
            priority=priority
        )


if __name__ == "__main__":
    # Example usage
    notifier = SignalNotifier()
    
    # Add a subscriber
    notifier.add_subscriber(email="user@example.com", phone="+86-1234567890", telegram_id="123456789")
    
    # Add some example signals
    notifier.add_signal(
        symbol="SH600519",
        name="贵州茅台",
        signal_type="BUY",
        strategy="MA_Cross",
        price=1800.50,
        reason="短期均线向上穿越长期均线",
        priority=3
    )
    
    notifier.add_signal(
        symbol="SZ000858",
        name="五粮液",
        signal_type="SELL",
        strategy="RSI",
        price=220.30,
        reason="RSI指标进入超买区域",
        priority=2
    )
    
    # Display recent signals
    print("\n📋 最近信号:")
    recent_signals = notifier.get_recent_signals(5)
    for signal in recent_signals:
        print(f"- {signal['timestamp']}: {signal['symbol']} {signal['signal_type']} ({signal['strategy']})")
    
    # Start monitoring
    notifier.monitor_signals_continuously()
    
    print("\n✅ 信号通知系统初始化完成")