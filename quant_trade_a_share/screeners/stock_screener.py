"""
Stock Screener for A-Share Market
Identifies potentially profitable stocks based on various criteria
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import the new EastMoney data fetcher and Ashare data fetcher
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.eastmoney_data_fetcher import EastMoneyDataFetcher
from data.ashare_data_fetcher import AshareDataFetcher


class StockScreener:
    def __init__(self, tushare_token=None):
        self.chinese_stocks = None
        self.screened_stocks = None
        # 仅用于回测
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
        # 主要数据源现在是EastMoney
        self.eastmoney_fetcher = EastMoneyDataFetcher()
        # 添加Ashare数据源
        self.ashare_fetcher = AshareDataFetcher()
    
    def get_chinese_stocks_list(self):
        """
        获取A股股票列表
        """
        try:
            # 使用新的EastMoney数据获取器
            all_stocks = self.eastmoney_fetcher.get_all_stocks()

            self.chinese_stocks = all_stocks[['symbol', 'code', 'name']]
            return self.chinese_stocks

        except Exception as e:
            print(f"获取股票列表失败: {e}")
            # 返回模拟数据用于测试
            return self._get_sample_stocks()
    
    def _get_sample_stocks(self):
        """
        获取样本股票数据用于测试
        """
        sample_data = {
            'symbol': ['sh600000', 'sz000001', 'sh600036', 'sz000858', 'sh600519', 'sh601398', 'sh601318', 'sz000002', 'sh600030', 'sh600276'],
            'code': ['600000', '000001', '600036', '000858', '600519', '601398', '601318', '000002', '600030', '600276'],
            'name': ['浦发银行', '平安银行', '招商银行', '五粮液', '贵州茅台', '工商银行', '中国平安', '万科A', '中信证券', '恒瑞医药']
        }
        self.chinese_stocks = pd.DataFrame(sample_data)
        return self.chinese_stocks
    
    def fetch_stock_data(self, symbol, period='180', freq='D', use_tushare_only=False, data_source='auto'):
        """
        获取单个股票的历史数据
        use_tushare_only: 仅用于回测时强制使用Tushare数据
        data_source: 指定数据源 ('auto', 'eastmoney', 'ashare', 'tushare')
        """
        try:
            if use_tushare_only and self.pro:
                # 仅用于回测 - 强制使用Tushare
                if freq != 'D':
                    # 获取10分钟级别数据（回测专用）
                    try:
                        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                        end_date = datetime.now().strftime('%Y%m%d')

                        # 移除交易所前缀以获取标准代码
                        code = symbol[2:] if symbol.startswith(('sh', 'sz')) else symbol

                        df = self.pro.query('bar', ts_code=code, freq='10min', start_date=start_date, end_date=end_date)

                        if df is not None and not df.empty:
                            # 重命名列以匹配标准格式
                            df.rename(columns={
                                'trade_time': 'date',
                                'ts_code': 'symbol',
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'vol': 'volume'
                            }, inplace=True)

                            # 转换日期格式
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)

                            # 计算技术指标
                            df = self._calculate_technical_indicators(df)

                            return df
                        else:
                            print(f"Tushare 未返回 {symbol} 的10分钟级别数据，可能需要高级权限")
                    except Exception as e:
                        print(f"回测数据获取失败: {e}")
                else:
                    # 获取日级别数据（回测专用）
                    try:
                        start_date = (datetime.now() - timedelta(days=int(period))).strftime('%Y%m%d')
                        end_date = datetime.now().strftime('%Y%m%d')

                        # 移除交易所前缀以获取标准代码
                        code = symbol[2:] if symbol.startswith(('sh', 'sz')) else symbol

                        df = self.pro.query('daily', ts_code=code, start_date=start_date, end_date=end_date)

                        if df is not None and not df.empty:
                            # 重命名列以匹配标准格式
                            df.rename(columns={
                                'trade_date': 'date',
                                'ts_code': 'symbol',
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'vol': 'volume'
                            }, inplace=True)

                            # 转换日期格式
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)

                            # 计算技术指标
                            df = self._calculate_technical_indicators(df)

                            return df
                        else:
                            print(f"Tushare 未返回 {symbol} 的日级别回测数据，可能需要高级权限")
                    except Exception as e:
                        print(f"回测日数据获取失败: {e}")

            # 实时信号生成现在支持多种数据源
            if freq != 'D':
                # 10分钟级别数据 - 主要使用EastMoney或模拟数据
                print(f"实时信号生成: {symbol} 使用EastMoney或模拟数据")
                # EastMoney通常不提供10分钟级别数据, 所以使用模拟数据
                return self._generate_mock_data(symbol, period, freq)
            else:
                # 日级别数据 - 根据参数选择数据源
                df = None

                if data_source == 'ashare' or data_source == 'auto':
                    # 优先使用Ashare
                    df = self.ashare_fetcher.fetch_stock_data(symbol, days=int(period))
                    if df is not None and not df.empty:
                        print(f"✅ 使用Ashare获取 {symbol} 数据")
                        df = self._calculate_technical_indicators(df)
                        return df

                if data_source == 'eastmoney' or (data_source == 'auto' and df is None):
                    # 尝试使用EastMoney作为备选
                    df = self.eastmoney_fetcher.fetch_stock_data(symbol, days=int(period))
                    if df is not None and not df.empty:
                        print(f"✅ 使用EastMoney获取 {symbol} 数据")
                        df = self._calculate_technical_indicators(df)
                        return df

                if data_source == 'tushare' or (data_source == 'auto' and df is None and self.pro):
                    # 如果EastMoney和Ashare都失败，尝试Tushare
                    try:
                        start_date = (datetime.now() - timedelta(days=int(period))).strftime('%Y%m%d')
                        end_date = datetime.now().strftime('%Y%m%d')

                        # 移除交易所前缀以获取标准代码
                        code = symbol[2:] if symbol.startswith(('sh', 'sz')) else symbol

                        df = self.pro.query('daily', ts_code=code, start_date=start_date, end_date=end_date)
                        if df is not None and not df.empty:
                            df.rename(columns={
                                'trade_date': 'date',
                                'ts_code': 'symbol',
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'vol': 'volume'
                            }, inplace=True)
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                            df = self._calculate_technical_indicators(df)
                            print(f"✅ 使用Tushare获取 {symbol} 数据")
                            return df
                    except Exception as e2:
                        print(f"Tushare备选方案失败: {e2}")

                # 如果所有真实数据源都失败，生成模拟数据作为最后的备选方案
                print(f"⚠️  所有真实数据源均不可用，为 {symbol} 生成模拟数据")
                return self._generate_mock_data(symbol, period, freq)
        except Exception as e:
            print(f"❌ 获取股票 {symbol} 数据失败: {e}")
            # 返回None表示获取失败
            return None
    
    def _generate_mock_data(self, symbol, period='180', freq='D'):
        """
        生成模拟股票数据
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        if freq != 'D':
            # 生成10分钟级别数据
            # 计算需要多少个10分钟间隔
            total_minutes = 240  # 4小时交易时间
            intervals = total_minutes // 10  # 24个10分钟间隔
            n = intervals  # 一天的数据点
            
            # 生成10分钟频率的时间索引
            end_time = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)  # A股收盘时间
            start_time = end_time - timedelta(minutes=240)  # 开盘时间
            
            # 生成10分钟间隔的时间序列
            dates = pd.date_range(start=start_time, end=end_time, freq='10T')
            # 只保留交易时间段内的数据
            dates = dates[(dates.hour >= 9) | (dates.hour < 15)]  # 9:30-15:00
            dates = dates[dates.hour != 12]  # 排除中午休市时间
            dates = dates[:n]  # 只取需要的数量
            
            if len(dates) == 0:
                return pd.DataFrame()
        else:
            # 生成日级别数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(period))
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # 只保留工作日
            dates = dates[dates.weekday < 5]
            
            n = len(dates)
            if n == 0:
                return pd.DataFrame()
        
        # 设置随机种子以确保结果一致
        np.random.seed(abs(hash(symbol)) % (2**32))
        
        if freq != 'D':
            # 10分钟级别数据 - 更高的波动性
            initial_price = 50 + np.random.random() * 100  # 初始价格在50-150之间
            returns = np.random.normal(0.0001, 0.005, n)  # 10分钟收益率均值0.01%，标准差0.5%
        else:
            # 日级别数据
            initial_price = 50 + np.random.random() * 100  # 初始价格在50-150之间
            returns = np.random.normal(0.0005, 0.02, n)  # 日收益率均值0.05%，标准差2%
        
        prices = [initial_price]
        
        for i in range(1, n):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.1))  # 确保价格为正
        
        prices = np.array(prices)
        
        # 生成OHLC数据
        # 开盘价通常是前一天收盘价加上一个小的随机变化
        opens = [prices[0]]
        for i in range(1, n):
            if freq != 'D':
                # 10分钟级别数据的小幅波动
                change = np.random.uniform(-0.002, 0.002)  # ±0.2%的变化
            else:
                change = np.random.uniform(-0.01, 0.01)  # ±1%的变化
            opens.append(prices[i-1] * (1 + change))
        opens = np.array(opens)
        
        # 高价、低价基于价格波动生成
        if freq != 'D':
            high_mult = np.random.uniform(1.0, 1.008, n)  # 10分钟级别较小波动
            low_mult = np.random.uniform(0.992, 1.0, n)
        else:
            high_mult = np.random.uniform(1.0, 1.03, n)
            low_mult = np.random.uniform(0.97, 1.0, n)
            
        highs = prices * high_mult
        lows = np.maximum(prices * low_mult, 0.1)  # 确保为正值
        closes = prices
        
        # 调整开盘价使其在高低区间内
        opens = np.clip(opens, lows, highs)
        
        # 生成成交量 (随价格变动而变动)
        if freq != 'D':
            # 10分钟级别数据的成交量更小
            base_volume = 10000 + np.random.random() * 40000  # 10k-50k股基础成交量
            volume_changes = np.abs(np.diff(closes, prepend=closes[0])) * 5000  # 成交量与价格变动相关
        else:
            base_volume = 1000000 + np.random.random() * 4000000  # 1M-5M股基础成交量
            volume_changes = np.abs(np.diff(closes, prepend=closes[0])) * 50000  # 成交量与价格变动相关
            
        volumes = (base_volume + volume_changes).astype(int)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates[:len(prices)],  # 确保长度一致
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        df.set_index('date', inplace=True)
        
        # 计算技术指标
        df = self._calculate_technical_indicators(df)
        
        if freq != 'D':
            print(f"⚠️  {symbol} 使用10分钟级别模拟数据")
        else:
            print(f"⚠️  {symbol} 使用日级别模拟数据")
        return df
    
    def _calculate_technical_indicators(self, df):
        """
        计算技术指标
        现在使用来自EastMoneyDataFetcher的增强指标
        """
        # 使用来自EastMoneyDataFetcher的增强指标
        return self.eastmoney_fetcher.calculate_enhanced_technical_indicators(df)
    
    def screen_stocks(self, filters=None):
        """
        根据筛选条件筛选股票
        
        参数:
        - filters: 筛选条件字典
        """
        if self.chinese_stocks is None:
            self.get_chinese_stocks_list()
        
        results = []
        
        # 默认筛选条件
        default_filters = {
            'min_price': 5,      # 最低价格
            'max_price': 200,    # 最高价格
            'min_volume': 1000000,  # 最小成交量
            'days_back': 90,     # 分析最近天数
            'min_return': 0.05,  # 最低收益率阈值
            'max_volatility': 0.05  # 最大波动率阈值
        }
        
        if filters:
            default_filters.update(filters)
        
        # 遍历股票进行筛选
        for idx, stock in self.chinese_stocks.iterrows():  # 限制数量以提高性能
            symbol = stock['symbol']
            name = stock['name']
            
            # 获取股票数据
            data = self.fetch_stock_data(symbol, period=str(default_filters['days_back']))
            
            if data is None or data.empty:
                continue
            
            # 计算基本面指标
            current_price = data['close'].iloc[-1]
            price_change_pct = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            avg_volume = data['volume'].mean()
            current_volatility = data['volatility'].iloc[-1] if not pd.isna(data['volatility'].iloc[-1]) else 0
            print(current_price,price_change_pct,avg_volume,current_volatility)
            # 应用筛选条件
            if (current_price >= default_filters['min_price'] and 
                current_price <= default_filters['max_price'] and
                avg_volume >= default_filters['min_volume'] and
                price_change_pct >= default_filters['min_return'] and
                current_volatility <= default_filters['max_volatility']):
                
                # 计算更多分析指标
                rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
                macd_histogram = data['histogram'].iloc[-1] if not pd.isna(data['histogram'].iloc[-1]) else 0
                price_pos = data['price_position'].iloc[-1] if not pd.isna(data['price_position'].iloc[-1]) else 0.5
                
                # 评估股票潜力分数
                score = self._calculate_potential_score(
                    current_price, price_change_pct, current_volatility, rsi, macd_histogram, price_pos
                )
                
                results.append({
                    'symbol': symbol,
                    'code': stock['code'],
                    'name': name,
                    'current_price': round(current_price, 2),
                    'price_change_pct': round(price_change_pct * 100, 2),
                    'avg_volume': int(avg_volume),
                    'volatility': round(current_volatility, 4),
                    'rsi': round(rsi, 2),
                    'score': round(score, 2)
                })
        
        # 按潜力分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        print(results)
        self.screened_stocks = pd.DataFrame(results)
        
        return self.screened_stocks
    
    def _calculate_potential_score(self, price, return_pct, volatility, rsi, macd_hist, price_pos):
        """
        计算股票潜力分数
        """
        # 基于多个因素计算综合得分
        score = 0

        # 收益率得分 (越高越好，但不过度)
        if return_pct > 20:
            score += 20
        elif return_pct > 0:
            score += return_pct
        else:
            score += return_pct * 0.5  # 负收益扣分较轻

        # 波动率得分 (适中最佳)
        if 0.02 <= volatility <= 0.04:
            score += 15
        elif 0.01 <= volatility <= 0.06:
            score += 10
        else:
            score += 5

        # RSI得分 (接近中间区域最佳)
        if 40 <= rsi <= 60:
            score += 15
        elif 30 <= rsi <= 70:
            score += 10
        else:
            score += 5

        # MACD柱状图得分
        if macd_hist > 0:
            score += min(macd_hist * 100, 10)  # 正向趋势加分
        else:
            score += max(macd_hist * 50, -10)  # 负向趋势扣分

        # 价格位置得分 (不处于极端位置最佳)
        if 0.2 <= price_pos <= 0.8:
            score += 10
        else:
            score += 5

        return score

    def get_top_active_stocks(self, limit=20):
        """
        获取最活跃的股票列表
        
        Args:
            limit (int): 返回的股票数量限制
            
        Returns:
            list: 包含股票信息的列表 [(symbol, name, volume), ...]
        """
        try:
            # 获取所有股票列表
            all_stocks = self.get_chinese_stocks_list()
            
            if all_stocks is None or all_stocks.empty:
                print("⚠️  无法获取股票列表")
                return []
            
            # 选择前limit只股票作为活跃股票（实际应用中应根据成交量等指标排序）
            active_stocks = []
            
            for idx, stock in all_stocks.head(limit).iterrows():
                symbol = stock['symbol']
                name = stock['name']
                
                # 获取该股票的成交量数据
                data = self.fetch_stock_data(symbol, period='30')  # 获取最近30天数据
                
                if data is not None and not data.empty and 'volume' in data.columns:
                    avg_volume = data['volume'].mean()
                else:
                    avg_volume = 0  # 如果无法获取数据，则设为0
                    
                active_stocks.append((symbol, name, avg_volume))
            
            # 按平均成交量降序排列
            active_stocks.sort(key=lambda x: x[2], reverse=True)
            
            return active_stocks[:limit]
            
        except Exception as e:
            print(f"❌ 获取活跃股票列表失败: {e}")
            # 返回一些示例数据作为备选
            sample_stocks = [
                ('sh600519', '贵州茅台', 10000000),
                ('sz000858', '五粮液', 8000000),
                ('sh600036', '招商银行', 15000000),
                ('sh601398', '工商银行', 20000000),
                ('sh601318', '中国平安', 12000000)
            ]
            return sample_stocks[:limit]


if __name__ == "__main__":
    screener = StockScreener()
    
    print("正在筛选有潜力的股票...")
    results = screener.screen_stocks({
        'min_price': 10,
        'max_price': 150,
        'min_volume': 5000000,
        'days_back': 60,
        'min_return': 0.02,
        'max_volatility': 0.04
    })
    
    if not results.empty:
        print("\n筛选结果 (按潜力分数排序):")
        print(results.head(10).to_string(index=False))
    else:
        print("未找到符合条件的股票")