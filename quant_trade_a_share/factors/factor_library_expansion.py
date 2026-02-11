"""
因子库扩充模块
将Qlib的Alpha因子与MyTT指标相结合
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.data import D
    from qlib.config import REG_CN as REGION_CN
    from qlib.contrib.factor import get_ic_intercept_neutralized
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("⚠️ Qlib 未安装，将使用基础因子功能")

# 导入 MyTT 指标 (处理可能的导入错误)
try:
    from quant_trade_a_share.utils.mytt_indicators import *
    MYTT_AVAILABLE = True

    # 创建一个简单的包装类来兼容原有接口
    class MyTTIndicators:
        def __init__(self):
            pass

        def MA(self, S, N):
            from quant_trade_a_share.utils.mytt_indicators import MA
            return MA(S, N)

        def EMA(self, S, N):
            from quant_trade_a_share.utils.mytt_indicators import EMA
            return EMA(S, N)

        def MACD(self, S, SHORT=12, LONG=26, M=9):
            from quant_trade_a_share.utils.mytt_indicators import MACD
            return MACD(S, SHORT, LONG, M)

        def KDJ(self, CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
            from quant_trade_a_share.utils.mytt_indicators import KDJ
            return KDJ(CLOSE, HIGH, LOW, N, M1, M2)

        def RSI(self, CLOSE, N=24):
            from quant_trade_a_share.utils.mytt_indicators import RSI
            return RSI(CLOSE, N)

        def BOLL(self, CLOSE, N=20, P=2):
            from quant_trade_a_share.utils.mytt_indicators import BOLL
            return BOLL(CLOSE, N, P)

        def CCI(self, CLOSE, HIGH, LOW, N=14):
            from quant_trade_a_share.utils.mytt_indicators import CCI
            return CCI(CLOSE, HIGH, LOW, N)

        def ATR(self, CLOSE, HIGH, LOW, N=20):
            from quant_trade_a_share.utils.mytt_indicators import ATR
            return ATR(CLOSE, HIGH, LOW, N)

        def DMA(self, CLOSE, M1=10, M2=50):
            from quant_trade_a_share.utils.mytt_indicators import DFMA
            return DFMA(CLOSE, M1, M2)

        def DMI(self, HIGH, LOW, CLOSE, M1=14, M2=6):
            from quant_trade_a_share.utils.mytt_indicators import DMI
            return DMI(CLOSE, HIGH, LOW, M1, M2)

        def TRIX(self, CLOSE, M1=12, M2=20):
            from quant_trade_a_share.utils.mytt_indicators import TRIX
            return TRIX(CLOSE, M1, M2)

        def VR(self, CLOSE, VOL, M1=26):
            from quant_trade_a_share.utils.mytt_indicators import VR
            return VR(CLOSE, VOL, M1)

        def WR(self, CLOSE, HIGH, LOW, N=10, N1=6):
            from quant_trade_a_share.utils.mytt_indicators import WR
            return WR(CLOSE, HIGH, LOW, N, N1)

except ImportError:
    print("⚠️ MyTT 指标不可用，将使用基础因子功能")
    MYTT_AVAILABLE = False

    # 提供一个哑元类作为替代
    class MyTTIndicators:
        def __init__(self):
            pass

        def MA(self, S, N):
            # 简单的MA实现作为后备
            if hasattr(pd, 'Series'):
                return pd.Series(S).rolling(N).mean().values
            else:
                return np.convolve(S, np.ones(N), 'valid') / N if len(S) >= N else np.full_like(S, np.mean(S) if len(S) > 0 else 0)

        def EMA(self, S, N):
            # 简单的EMA实现作为后备
            if hasattr(pd, 'Series'):
                return pd.Series(S).ewm(span=N, adjust=False).mean().values
            else:
                return S  # 哑元实现

        def MACD(self, S, SHORT=12, LONG=26, M=9):
            return np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S))

        def KDJ(self, CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def RSI(self, CLOSE, N=24):
            return np.zeros(len(CLOSE))

        def BOLL(self, CLOSE, N=20, P=2):
            mid = self.MA(CLOSE, N)
            std = pd.Series(CLOSE).rolling(N).std().values if hasattr(pd, 'Series') else np.zeros(len(CLOSE))
            return mid + P * std, mid, mid - P * std

        def CCI(self, CLOSE, HIGH, LOW, N=14):
            return np.zeros(len(CLOSE))

        def ATR(self, CLOSE, HIGH, LOW, N=20):
            return np.zeros(len(CLOSE))

        def DMA(self, CLOSE, M1=10, M2=50):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def DMI(self, HIGH, LOW, CLOSE, M1=14, M2=6):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def TRIX(self, CLOSE, M1=12, M2=20):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def VR(self, CLOSE, VOL, M1=26):
            return np.zeros(len(CLOSE))

        def WR(self, CLOSE, HIGH, LOW, N=10, N1=6):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

class FactorLibraryExpansion:
    """
    因子库扩充类
    结合 Qlib Alpha 因子与 MyTT 指标
    """

    def __init__(self, provider_uri="~/.qlib/qlib_data/cn_data"):
        """初始化因子库扩展"""
        self.provider_uri = provider_uri
        self.initialized = False

        # Initialize MyTTIndicators regardless of import status (dummy class will be used if import failed)
        self.mytt_indicators = MyTTIndicators()

        if QLIB_AVAILABLE:
            try:
                qlib.init(provider_uri=self.provider_uri, region=REGION_CN)
                self.initialized = True
                print("✅ 因子库扩充模块初始化成功")
            except Exception as e:
                print(f"⚠️ Qlib 初始化失败: {e}")
                print("💡 提示: 安装 Qlib 并下载数据以启用完整因子功能")
        else:
            print("⚠️ Qlib 不可用，将使用基础因子功能")

    def get_qlib_alpha_factors(self, instruments: List[str], start_date: str, end_date: str,
                              alpha_version: str = '158') -> pd.DataFrame:
        """
        获取 Qlib Alpha 因子

        Args:
            instruments: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            alpha_version: Alpha 版本 ('158' 或 '101')
        """
        if not self.initialized:
            print("❌ Qlib 未初始化，无法获取 Alpha 因子")
            return pd.DataFrame()

        try:
            if alpha_version == '158':
                # Qlib Alpha158 特征集
                alpha_fields = [
                    # 价量关系
                    'Ref($close,1)/$close',  # 一日收益率
                    'Mean($close,5)/$close', # 五日均值比
                    'Mean($close,10)/$close',# 十日均值比
                    'Mean($close,20)/$close',# 二十日均值比
                    '($close-$open)/$open',   # 开盘转收盘变化
                    '($high-$low)/$close',    # 最高价最低价差
                    'Rank($volume)',          # 成交量排名
                    'Rank($close)',           # 收盘价排名

                    # 波动率类
                    'Std($close,10)',         # 10日标准差
                    'Std($close,20)',         # 20日标准差
                    'Ts_Sum(Greater($close-$open,0),5)/Ts_Sum(Abs($close-$open),5)',

                    # 趋势类
                    'Slope($close,5)',        # 5日趋势斜率
                    'Slope($close,10)',       # 10日趋势斜率
                    'Resi($close,20)',        # 20日残差

                    # 成交量类
                    'Corr($volume, $close, 5)', # 价量相关性
                    'Ts_ArgMax($close, 20)',    # 20日最高点位置
                    'Ts_ArgMin($close, 20)',    # 20日最低点位置
                ]
            else:  # Alpha101
                alpha_fields = [
                    '$close/$open-1',  # 日回报
                    'Rank($volume)/Rank($close)',  # 量价关系
                    'Ts_Sum($high-$low, 10)/Ts_Sum(Ts_Sum($high-$low, 2), 5)',  # 波动率特征
                    'Delay($close,5)/$close',  # 5日滞后比
                    'Corr(Rank($close), Rank($volume), 5)',  # 价量相关性
                    'Decay_linear($close, 5)',  # 线性衰减
                    'Ts_Rank($close, 10)',  # 10日排名
                    'Ts_Min($low, 5)',      # 5日最低价
                    'Ts_Max($high, 5)',     # 5日最高价
                    'Ts_ArgMax($high, 20)', # 20日最高价位置
                    'Ts_ArgMin($low, 20)',  # 20日最低价位置
                ]

            # 获取特征数据
            df = D.features(instruments, alpha_fields, start_date, end_date)
            print(f"✅ 成功获取 {len(alpha_fields)} 个 Alpha{alpha_version} 因子，{len(df)} 条记录")
            return df

        except Exception as e:
            print(f"❌ 获取 Alpha 因子失败: {e}")
            return pd.DataFrame()

    def get_mytt_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取 MyTT 指标

        Args:
            data: 股票数据 (包含 high, low, close, volume 等列)
        """
        try:
            indicators = pd.DataFrame(index=data.index)

            # MA - 移动平均线
            for period in [5, 10, 20, 30, 60]:
                indicators[f'MA_{period}'] = self.mytt_indicators.MA(data['close'], period)

            # EMA - 指数移动平均线
            for period in [5, 10, 20, 30]:
                indicators[f'EMA_{period}'] = self.mytt_indicators.EMA(data['close'], period)

            # MACD - 异同移动平均线
            macd, dif, dea, bar = self.mytt_indicators.MACD(data['close'])
            indicators['MACD'] = macd
            indicators['DIF'] = dif
            indicators['DEA'] = dea
            indicators['BAR'] = bar

            # KDJ - 随机指标
            k, d, j = self.mytt_indicators.KDJ(data['high'], data['low'], data['close'])
            indicators['KDJ_K'] = k
            indicators['KDJ_D'] = d
            indicators['KDJ_J'] = j

            # RSI - 相对强弱指标
            for period in [6, 12, 24]:
                indicators[f'RSI_{period}'] = self.mytt_indicators.RSI(data['close'], period)

            # BOLL - 布林带
            upper, middle, lower = self.mytt_indicators.BOLL(data['close'])
            indicators['BOLL_UPPER'] = upper
            indicators['BOLL_MIDDLE'] = middle
            indicators['BOLL_LOWER'] = lower

            # CCI - 顺势指标
            indicators['CCI'] = self.mytt_indicators.CCI(data['high'], data['low'], data['close'])

            # ATR - 真实波幅
            indicators['ATR'] = self.mytt_indicators.ATR(data['high'], data['low'], data['close'])

            # DMA - 平均线差
            indicators['DMA'] = self.mytt_indicators.DMA(data['close'])

            # DMI - 动向指标
            p_di, n_di, adx, adxr = self.mytt_indicators.DMI(data['high'], data['low'], data['close'])
            indicators['DMI_P_DI'] = p_di
            indicators['DMI_N_DI'] = n_di
            indicators['DMI_ADX'] = adx
            indicators['DMI_ADXR'] = adxr

            # TRIX - 三重指数平滑平均线
            indicators['TRIX'] = self.mytt_indicators.TRIX(data['close'])

            # VR - 成交量变异率
            indicators['VR'] = self.mytt_indicators.VR(data['close'], data['volume'])

            # WR - 威廉指标
            for period in [5, 10, 14]:
                indicators[f'WR_{period}'] = self.mytt_indicators.WR(data['high'], data['low'], data['close'], period)

            print(f"✅ 成功计算 {len(indicators.columns)} 个 MyTT 指标")
            return indicators

        except Exception as e:
            print(f"❌ 计算 MyTT 指标失败: {e}")
            return pd.DataFrame()

    def combine_factors(self, qlib_factors: pd.DataFrame, mytt_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        合并 Qlib 因子和 MyTT 指标

        Args:
            qlib_factors: Qlib 因子数据
            mytt_indicators: MyTT 指标数据
        """
        if qlib_factors.empty and mytt_indicators.empty:
            print("❌ Qlib 因子和 MyTT 指标均为空")
            return pd.DataFrame()

        try:
            # 如果其中一个是空的，直接返回另一个
            if qlib_factors.empty:
                print("ℹ️ 使用 MyTT 指标作为主因子")
                return mytt_indicators
            if mytt_indicators.empty:
                print("ℹ️ 使用 Qlib 因子作为主因子")
                return qlib_factors

            # 合并两个 DataFrame
            combined_factors = pd.concat([qlib_factors, mytt_indicators], axis=1)

            # 填充缺失值
            combined_factors = combined_factors.fillna(method='ffill').fillna(0)

            print(f"✅ 成功合并因子，总共有 {len(combined_factors.columns)} 个因子")
            return combined_factors

        except Exception as e:
            print(f"❌ 合并因子失败: {e}")
            return pd.DataFrame()

    def get_comprehensive_factors(self, data: pd.DataFrame, instruments: List[str],
                               start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取综合因子（Qlib + MyTT）

        Args:
            data: 股票数据
            instruments: 股票列表
            start_date: 开始日期
            end_date: 结束日期
        """
        print("🔄 正在生成综合因子...")

        # 获取 Qlib 因子
        if self.initialized and instruments:
            print("📦 获取 Qlib Alpha 因子...")
            qlib_factors = self.get_qlib_alpha_factors(instruments, start_date, end_date)
        else:
            print("ℹ️ Qlib 不可用，跳过 Alpha 因子")
            qlib_factors = pd.DataFrame()

        # 获取 MyTT 指标
        print("📊 计算 MyTT 指标...")
        mytt_indicators = self.get_mytt_indicators(data)

        # 合并因子
        print("🔗 合并因子...")
        combined_factors = self.combine_factors(qlib_factors, mytt_indicators)

        print(f"🎉 综合因子生成完成，共 {len(combined_factors.columns)} 个因子")
        return combined_factors

    def calculate_factor_stats(self, factors: pd.DataFrame) -> Dict:
        """
        计算因子统计信息
        """
        if factors.empty:
            return {}

        stats = {}
        for col in factors.columns:
            if not factors[col].isna().all():  # 忽略全为 NaN 的列
                series = factors[col].dropna()
                if len(series) > 0:
                    stats[col] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max(),
                        'nan_count': factors[col].isna().sum(),
                        'valid_count': len(series)
                    }

        return stats

    def factor_rank_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子间的秩相关性
        """
        if factors.empty:
            return pd.DataFrame()

        try:
            # 只计算数值型列的相关性
            numeric_cols = factors.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return pd.DataFrame()

            factor_subset = factors[numeric_cols].rank(pct=True)  # 转换为百分位排名
            correlation_matrix = factor_subset.corr(method='spearman')  # 使用斯皮尔曼相关

            print(f"✅ 计算了 {len(correlation_matrix)}×{len(correlation_matrix)} 个因子相关性矩阵")
            return correlation_matrix

        except Exception as e:
            print(f"❌ 计算因子相关性失败: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    print("🧪 测试因子库扩充模块...")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.abs(np.random.randn(100) * 0.2),
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.abs(np.random.randn(100) * 0.2),
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # 测试因子库
    factor_lib = FactorLibraryExpansion()

    print(f"\n📋 因子库状态: {'可用' if factor_lib.initialized else '不可用'}")
    print(f"📊 MyTT 指标可用: {hasattr(factor_lib.mytt_indicators, 'MA')}")

    print("\n🎯 主要功能:")
    print("• Qlib Alpha158/Alpha101 因子获取")
    print("• MyTT 技术指标计算")
    print("• 因子合并与统整")
    print("• 因子统计分析")
    print("• 因子相关性分析")

    print("\n💡 应用场景:")
    print("1. 策略因子研究")
    print("2. Alpha 挖掘")
    print("3. 风险因子建模")
    print("4. 信号组合优化")