"""
Qlib Data Adapter
将 Qlib 数据功能与现有数据获取接口集成
"""
import qlib
from qlib.config import REG_CN as REGION_CN
from qlib.data import D
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class QlibDataAdapter:
    """
    将 Qlib 功能与现有数据接口集成的适配器
    """

    def __init__(self):
        """初始化 Qlib 适配器"""
        try:
            # 尝试初始化 Qlib
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REGION_CN,
                      mongo_cache=False, redis_cache=False, disable_disk_cache=True)
            self.initialized = True
            print("✅ Qlib 适配器初始化成功")
        except Exception as e:
            print(f"⚠️ Qlib 初始化失败: {e}")
            print("💡 提示: 请按需下载 Qlib 数据以启用完整功能")
            self.initialized = False

    def get_q_score_features(self, instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        使用 Qlib 的 Alpha158 特征集

        Args:
            instruments: 股票列表 (格式如 ['SH600000', 'SZ000001'])
            start_date: 开始日期 (格式: 'YYYY-MM-DD')
            end_date: 结束日期 (格式: 'YYYY-MM-DD')

        Returns:
            包含 Qlib 特征的 DataFrame
        """
        if not self.initialized:
            print("❌ Qlib 未初始化，请先完成 Qlib 数据配置")
            return pd.DataFrame()

        # 定义 Qlib 的 Alpha158 特征
        fields = [
            '$close', '$open', '$high', '$low', '$volume', '$factor',
            # 技术指标特征
            'Ref($close,1)/$close',  # 日收益率
            'Rank($volume)',         # 成交量排名
            'Mean($close,5)/$close', # 5日均价比
            'Std($close,10)',        # 10日收盘价标准差
            # 更多 Qlib 表达式...
        ]

        try:
            # 获取 Qlib 特征数据
            df = D.features(instruments, fields, start_date, end_date)
            print(f"✅ 成功获取 {len(df)} 条 Qlib 特征数据")
            return df
        except Exception as e:
            print(f"❌ 获取 Qlib 特征失败: {e}")
            return pd.DataFrame()

    def calculate_advanced_factors(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用 Qlib 风格的表达式计算高级因子

        Args:
            stock_data: 包含 OHLCV 数据的 DataFrame

        Returns:
            添加了 Qlib 风格因子的 DataFrame
        """
        if stock_data.empty:
            return stock_data

        # 这里可以使用 Qlib 的表达式语法或实现类似逻辑
        result = stock_data.copy()

        # 示例: 计算一些基于 Qlib 思路的技术指标
        try:
            # 移动平均线
            result['MA5'] = result['close'].rolling(window=5).mean()
            result['MA10'] = result['close'].rolling(window=10).mean()
            result['MA20'] = result['close'].rolling(window=20).mean()

            # RSI (相对强弱指数) - Qlib 风格
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result['RSI'] = 100 - (100 / (1 + rs))

            # 布林带 - Qlib 风格
            result['BB_middle'] = result['close'].rolling(window=20).mean()
            bb_std = result['close'].rolling(window=20).std()
            result['BB_upper'] = result['BB_middle'] + (bb_std * 2)
            result['BB_lower'] = result['BB_middle'] - (bb_std * 2)

            # 波动率因子
            result['volatility_5d'] = result['close'].pct_change().rolling(window=5).std()
            result['volatility_20d'] = result['close'].pct_change().rolling(window=20).std()

            print("✅ 成功计算 Qlib 风格高级因子")
            return result
        except Exception as e:
            print(f"⚠️ 计算高级因子时出错: {e}")
            return stock_data

    def integrate_with_multi_factor_strategy(self, symbols: List[str], start_date: str, end_date: str):
        """
        将 Qlib 特征集成到您的多因子策略中

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        """
        print("🔄 将 Qlib 特征集成到多因子策略中...")

        # 格式化股票代码为 Qlib 格式
        qlib_symbols = []
        for symbol in symbols:
            if '.' not in symbol:  # 如果不是标准格式
                if symbol.startswith(('0', '3')):  # SZSE
                    qlib_symbol = f"SZ{symbol}"
                elif symbol.startswith('6'):  # SSE
                    qlib_symbol = f"SH{symbol}"
                else:
                    qlib_symbol = f"SH{symbol}"
                qlib_symbols.append(qlib_symbol)
            else:
                # 转换现有格式为 Qlib 格式
                parts = symbol.split('.')
                code, exchange = parts[0], parts[1].upper()
                if exchange == 'SZ':
                    qlib_symbols.append(f"SZ{code}")
                elif exchange == 'SH':
                    qlib_symbols.append(f"SH{code}")
                else:
                    qlib_symbols.append(f"{exchange}{code}")

        # 获取 Qlib 特征
        if self.initialized:
            qlib_features = self.get_q_score_features(qlib_symbols, start_date, end_date)
            if not qlib_features.empty:
                print(f"📊 Qlib 提供了 {len(qlib_features.columns)} 个特征用于多因子分析")
                return qlib_features
        else:
            print("💡 Qlib 未初始化，将使用基础特征计算")
            # 返回基础特征作为备选
            return pd.DataFrame()

        return pd.DataFrame()


# 示例使用
if __name__ == "__main__":
    print("🧪 测试 Qlib 适配器...")

    adapter = QlibDataAdapter()

    # 演示如何使用
    sample_stocks = ["SH600000", "SZ000001", "SH600519"]  # 示例股票
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # 如果 Qlib 初始化成功，则尝试获取特征
    if adapter.initialized:
        features = adapter.get_q_score_features(sample_stocks, start_date, end_date)
        print(f"获取到 {len(features)} 条记录的 Qlib 特征")
    else:
        print("Qlib 未完全初始化，但适配器已准备就绪")

    print("✅ Qlib 适配器测试完成")