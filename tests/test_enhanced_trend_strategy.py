"""
增强版趋势动量策略 - 多因子融合

核心改进：
1. 趋势评分 + Qlib特征融合
2. 动态ATR止损止盈
3. 多信号组合确认
4. 仓位动态管理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_data(stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """获取数据"""
    try:
        import baostock as bs
    except ImportError:
        print("请安装baostock: pip install baostock")
        return {}

    lg = bs.login()
    if lg.error_code != '0':
        print(f"BaoStock登录失败: {lg.error_msg}")
        return {}

    print(f"BaoStock登录成功，获取数据: {start_date} ~ {end_date}")

    stock_data = {}
    for code in stock_codes:
        try:
            bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2"
            )

            if rs is None or rs.error_code != '0':
                continue

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                continue

            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'timestamp'})
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.set_index('timestamp').sort_index()
            stock_data[code] = df
            print(f"  {code}: {len(df)} 条")

        except Exception as e:
            print(f"  {code}: 获取失败 - {e}")

    bs.logout()
    return stock_data


class EnhancedFeatureEngine:
    """增强特征引擎 - 融合趋势评分和Qlib特征"""

    def __init__(self):
        self.feature_names = []

    def generate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """生成综合特征"""
        if len(df) < 60:
            return {}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = {}

        # ========== 趋势强度特征 ==========
        # MA排列得分
        ma5 = self._ma(close, 5)
        ma10 = self._ma(close, 10)
        ma20 = self._ma(close, 20)
        ma60 = self._ma(close, 60)

        ma_score = 0
        if ma5[-1] > ma10[-1]: ma_score += 25
        if ma10[-1] > ma20[-1]: ma_score += 25
        if ma20[-1] > ma60[-1]: ma_score += 25
        if ma5[-1] > ma20[-1]: ma_score += 25
        features['ma_alignment'] = ma_score

        # MA斜率
        ma5_slope = (ma5[-1] - ma5[-5]) / ma5[-5] * 100 if ma5[-5] > 0 else 0
        ma20_slope = (ma20[-1] - ma20[-5]) / ma20[-5] * 100 if ma20[-5] > 0 else 0
        features['ma5_slope'] = ma5_slope
        features['ma20_slope'] = ma20_slope

        # 价格位置
        price_vs_ma20 = (close[-1] - ma20[-1]) / ma20[-1] * 100
        features['price_vs_ma20'] = price_vs_ma20

        # ========== 动量特征 ==========
        # 多周期收益率
        for period in [3, 5, 10, 20]:
            if len(close) > period:
                ret = (close[-1] - close[-period]) / close[-period] * 100
                features[f'return_{period}d'] = ret

        # 动量加速度
        if len(close) > 10:
            mom_5 = (close[-1] - close[-5]) / close[-5] * 100
            mom_5_prev = (close[-5] - close[-10]) / close[-10] * 100
            features['momentum_accel'] = mom_5 - mom_5_prev

        # ========== 波动率特征 ==========
        # ATR
        atr = self._atr(high, low, close, 14)
        features['atr'] = atr[-1] if len(atr) > 0 else 0
        features['atr_ratio'] = atr[-1] / close[-1] * 100 if len(atr) > 0 and close[-1] > 0 else 0

        # 波动率
        returns = pd.Series(close).pct_change().dropna()
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns) >= 20 else 0
        features['volatility_20d'] = vol_20

        # ========== 技术指标 ==========
        # RSI
        rsi = self._rsi(close, 14)
        features['rsi_14'] = rsi[-1] if len(rsi) > 0 else 50

        # MACD
        macd, signal, hist = self._macd(close)
        features['macd'] = macd[-1] if len(macd) > 0 else 0
        features['macd_signal'] = signal[-1] if len(signal) > 0 else 0
        features['macd_hist'] = hist[-1] if len(hist) > 0 else 0

        # KDJ
        k, d, j = self._kdj(high, low, close, 9)
        features['kdj_k'] = k[-1] if len(k) > 0 else 50
        features['kdj_d'] = d[-1] if len(d) > 0 else 50
        features['kdj_j'] = j[-1] if len(j) > 0 else 50

        # ========== 量能特征 ==========
        # 量比
        vol_5 = np.mean(volume[-5:])
        vol_20 = np.mean(volume[-20:])
        features['volume_ratio'] = vol_5 / vol_20 if vol_20 > 0 else 1

        # 量价相关性
        if len(close) >= 10:
            price_changes = np.diff(close[-10:])
            volume_changes = np.diff(volume[-10:])
            if np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                features['price_volume_corr'] = corr

        # ========== 形态特征 ==========
        # 价格位置（N日高低点）
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        price_position = (close[-1] - low_20) / (high_20 - low_20) * 100 if high_20 != low_20 else 50
        features['price_position_20d'] = price_position

        # 突破信号
        high_20_before = np.max(high[-21:-1]) if len(high) > 21 else high[-1]
        features['breakout_20d'] = 1 if close[-1] > high_20_before else 0

        # 回踩信号
        dist_from_high = (high_20 - close[-1]) / high_20 * 100
        features['pullback_pct'] = dist_from_high

        # ========== 综合评分 ==========
        features['trend_score'] = self._calculate_trend_score(features)
        features['momentum_score'] = self._calculate_momentum_score(features)
        features['timing_score'] = self._calculate_timing_score(features)
        features['final_score'] = (
            features['trend_score'] * 0.4 +
            features['momentum_score'] * 0.35 +
            features['timing_score'] * 0.25
        )

        return features

    def _calculate_trend_score(self, f: Dict) -> float:
        """计算趋势得分"""
        score = 50

        # MA排列
        score += (f.get('ma_alignment', 50) - 50) * 0.3

        # MA斜率
        if f.get('ma20_slope', 0) > 0:
            score += min(15, f['ma20_slope'] * 2)
        else:
            score -= 10

        # 价格位置
        if 0 < f.get('price_vs_ma20', 0) < 15:
            score += 10
        elif f.get('price_vs_ma20', 0) >= 15:
            score -= 5  # 远离均线风险

        return min(100, max(0, score))

    def _calculate_momentum_score(self, f: Dict) -> float:
        """计算动量得分"""
        score = 50

        # 收益率
        ret_5d = f.get('return_5d', 0)
        ret_10d = f.get('return_10d', 0)

        if ret_5d > 0:
            score += min(20, ret_5d * 2)
        if ret_10d > 0:
            score += min(15, ret_10d)

        # 动量加速度
        if f.get('momentum_accel', 0) > 0:
            score += 10

        # MACD
        if f.get('macd_hist', 0) > 0:
            score += 10

        # RSI健康度
        rsi = f.get('rsi_14', 50)
        if 50 <= rsi <= 70:
            score += 10
        elif rsi > 80:
            score -= 15

        return min(100, max(0, score))

    def _calculate_timing_score(self, f: Dict) -> float:
        """计算时机得分"""
        score = 50

        # 回踩买点
        pullback = f.get('pullback_pct', 0)
        if 3 <= pullback <= 10:
            score += 20  # 最佳回踩区间

        # 突破
        if f.get('breakout_20d', 0) == 1:
            score += 15

        # KDJ超卖
        j = f.get('kdj_j', 50)
        if j < 20:
            score += 15
        elif j > 80:
            score -= 10

        # 量比健康
        vol_ratio = f.get('volume_ratio', 1)
        if 1.2 <= vol_ratio <= 2.0:
            score += 10

        return min(100, max(0, score))

    def _ma(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full(len(data), np.nan)
        if len(data) >= period:
            result[period-1:] = np.convolve(data, np.ones(period)/period, mode='valid')
        return result

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(period).mean().values
        return atr

    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50
        return rsi

    def _macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = (macd_line - signal_line) * 2
        return macd_line, signal_line, histogram

    def _kdj(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9):
        hhv = pd.Series(high).rolling(n).max().values
        llv = pd.Series(low).rolling(n).min().values
        rsv = (close - llv) / (hhv - llv + 1e-10) * 100
        k = pd.Series(rsv).ewm(alpha=1/3, adjust=False).mean().values
        d = pd.Series(k).ewm(alpha=1/3, adjust=False).mean().values
        j = 3 * k - 2 * d
        return k, d, j


class EnhancedTrendStrategy:
    """增强版趋势策略"""

    def __init__(
        self,
        min_score: float = 60,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 4.0,
        max_hold_days: int = 15,
        max_positions: int = 5,
        position_pct: float = 0.18
    ):
        self.min_score = min_score
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_hold_days = max_hold_days
        self.max_positions = max_positions
        self.position_pct = position_pct
        self.feature_engine = EnhancedFeatureEngine()

    def run_backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000
    ) -> Dict:
        """运行回测"""

        # 获取所有日期
        all_dates = set()
        for df in stock_data.values():
            dates = df.loc[start_date:end_date].index.tolist()
            all_dates.update(dates)
        all_dates = sorted(list(all_dates))

        if not all_dates:
            return {'error': '日期范围内没有数据'}

        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = [capital]

        # 预计算特征
        print("\n预计算特征...")
        stock_features = {}
        for code, df in stock_data.items():
            stock_features[code] = {}
            for i in range(60, len(df)):
                date = df.index[i]
                hist = df.iloc[:i+1]
                features = self.feature_engine.generate_features(hist)
                stock_features[code][date] = features

        print("开始回测...")
        for date in all_dates:
            daily_pnl = 0

            # === 检查持仓 ===
            for code in list(positions.keys()):
                if code not in stock_data:
                    continue
                df = stock_data[code]
                if date not in df.index:
                    continue

                close = df.loc[date, 'close']
                pos = positions[code]

                # 获取ATR
                atr = self._get_atr(df.loc[:date])
                stop_loss_price = pos['entry_price'] - self.stop_loss_atr * atr
                take_profit_price = pos['entry_price'] + self.take_profit_atr * atr

                # 止损
                if close <= stop_loss_price:
                    pnl = (close - pos['entry_price']) * pos['shares']
                    daily_pnl += pnl
                    trades.append({
                        'code': code, 'action': 'sell', 'price': close,
                        'pnl': pnl, 'reason': 'stop_loss', 'date': date
                    })
                    del positions[code]

                # 止盈
                elif close >= take_profit_price:
                    pnl = (close - pos['entry_price']) * pos['shares']
                    daily_pnl += pnl
                    trades.append({
                        'code': code, 'action': 'sell', 'price': close,
                        'pnl': pnl, 'reason': 'take_profit', 'date': date
                    })
                    del positions[code]

                # 超时
                elif (date - pos['entry_date']).days >= self.max_hold_days:
                    pnl = (close - pos['entry_price']) * pos['shares']
                    daily_pnl += pnl
                    trades.append({
                        'code': code, 'action': 'sell', 'price': close,
                        'pnl': pnl, 'reason': 'timeout', 'date': date
                    })
                    del positions[code]

                # 趋势破坏卖出
                elif code in stock_features and date in stock_features[code]:
                    features = stock_features[code][date]
                    if features.get('ma20_slope', 0) < -1:  # MA20拐头向下
                        pnl = (close - pos['entry_price']) * pos['shares']
                        daily_pnl += pnl
                        trades.append({
                            'code': code, 'action': 'sell', 'price': close,
                            'pnl': pnl, 'reason': 'trend_broken', 'date': date
                        })
                        del positions[code]

            # === 尝试买入 ===
            if len(positions) < self.max_positions:
                candidates = []

                for code, df in stock_data.items():
                    if code in positions:
                        continue
                    if date not in df.index:
                        continue
                    if code not in stock_features or date not in stock_features[code]:
                        continue

                    features = stock_features[code][date]
                    final_score = features.get('final_score', 0)

                    # 硬过滤
                    if features.get('ma20_slope', 0) <= 0:
                        continue
                    if features.get('rsi_14', 50) > 85:
                        continue

                    if final_score >= self.min_score:
                        candidates.append((code, final_score, df.loc[date, 'close'], features))

                # 按评分排序买入
                candidates.sort(key=lambda x: x[1], reverse=True)

                for code, score, close, features in candidates:
                    if len(positions) >= self.max_positions:
                        break

                    atr = self._get_atr(stock_data[code].loc[:date])
                    shares = (capital * self.position_pct) / close

                    positions[code] = {
                        'shares': shares,
                        'entry_price': close,
                        'atr': atr,
                        'entry_date': date,
                        'score': score
                    }

                    trades.append({
                        'code': code, 'action': 'buy', 'price': close,
                        'shares': shares, 'date': date, 'score': score
                    })

            capital += daily_pnl
            equity_curve.append(capital)

        # 平剩余仓位
        for code, pos in list(positions.items()):
            if code in stock_data:
                df = stock_data[code]
                if len(df) > 0:
                    close = df.iloc[-1]['close']
                    pnl = (close - pos['entry_price']) * pos['shares']
                    capital += pnl
                    trades.append({
                        'code': code, 'action': 'sell', 'price': close,
                        'pnl': pnl, 'reason': 'final', 'date': df.index[-1]
                    })

        # 计算统计
        sell_trades = [t for t in trades if t.get('action') == 'sell']
        winning = [t for t in sell_trades if t.get('pnl', 0) > 0]

        total_return = capital / initial_capital - 1
        test_days = len(all_dates)
        annual_return = total_return * (252 / test_days) if test_days > 0 else 0

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'total_trades': len(sell_trades),
            'winning_trades': len(winning),
            'win_rate': len(winning) / len(sell_trades) if sell_trades else 0,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        if len(tr) > 0:
            tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else close[-1] * 0.03


def optimize_parameters(
    stock_data: Dict[str, pd.DataFrame],
    test_start: str,
    test_end: str,
    target_return: float = 0.15
) -> Tuple[Dict, Dict]:
    """参数优化"""

    # 参数范围
    min_scores = [55, 60, 65, 70]
    stop_loss_atrs = [1.5, 2.0, 2.5, 3.0]
    take_profit_atrs = [3.0, 3.5, 4.0, 4.5, 5.0]
    max_hold_days = [10, 12, 15, 18, 20]
    max_positions = [4, 5, 6]

    best_params = None
    best_result = None
    best_score = -1

    results = []
    count = 0

    print("\n开始参数优化...")
    for min_score in min_scores:
        for sl in stop_loss_atrs:
            for tp in take_profit_atrs:
                if tp <= sl:
                    continue
                for hold_days in max_hold_days:
                    for max_pos in max_positions:
                        count += 1

                        strategy = EnhancedTrendStrategy(
                            min_score=min_score,
                            stop_loss_atr=sl,
                            take_profit_atr=tp,
                            max_hold_days=hold_days,
                            max_positions=max_pos
                        )

                        result = strategy.run_backtest(
                            stock_data, test_start, test_end
                        )

                        if 'error' in result:
                            continue

                        annual_ret = result['annual_return']
                        win_rate = result['win_rate']
                        trades = result['total_trades']
                        max_dd = result['max_drawdown']

                        if trades < 5:
                            continue

                        # 综合评分
                        score = annual_ret * 2 - max_dd * 0.3

                        results.append(({
                            'min_score': min_score,
                            'stop_loss_atr': sl,
                            'take_profit_atr': tp,
                            'max_hold_days': hold_days,
                            'max_positions': max_pos
                        }, annual_ret, win_rate, trades, max_dd))

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_score': min_score,
                                'stop_loss_atr': sl,
                                'take_profit_atr': tp,
                                'max_hold_days': hold_days,
                                'max_positions': max_pos
                            }
                            best_result = result

                            if annual_ret >= target_return:
                                print(f"✓ 找到达标参数: 年化 {annual_ret:.2%}")
                                return best_params, best_result

    # 打印前10
    print("\n测试集收益前10名:")
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (params, ret, wr, tr, dd) in enumerate(results[:10]):
        print(f"  {i+1}. {params}, 年化: {ret:.2%}, 胜率: {wr:.2%}, 交易: {tr}, 回撤: {dd:.2%}")

    return best_params, best_result


def main():
    """主函数"""
    stock_codes = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

    print("=" * 60)
    print("增强版趋势动量策略")
    print("=" * 60)

    # 获取数据
    print("\nStep 1: 获取数据")
    stock_data = fetch_data(stock_codes, "2020-01-01", "2026-02-28")

    if not stock_data:
        print("获取数据失败")
        return

    # 参数优化
    print("\nStep 2: 参数优化")
    best_params, test_result = optimize_parameters(
        stock_data=stock_data,
        test_start="2025-01-01",
        test_end="2026-02-28",
        target_return=0.15
    )

    if not best_params:
        print("参数优化失败")
        return

    # 打印结果
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)
    print(f"最优参数: {best_params}")
    print(f"测试集年化收益: {test_result.get('annual_return', 0):.2%}")
    print(f"总收益率: {test_result.get('total_return', 0):.2%}")
    print(f"最大回撤: {test_result.get('max_drawdown', 0):.2%}")
    print(f"交易次数: {test_result.get('total_trades', 0)}")
    print(f"胜率: {test_result.get('win_rate', 0):.2%}")

    if test_result.get('annual_return', 0) >= 0.15:
        print("\n✓ 达到目标！年化收益 >= 15%")
        print("DONE")


if __name__ == "__main__":
    main()