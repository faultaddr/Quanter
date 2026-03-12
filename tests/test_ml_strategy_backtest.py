"""
机器学习选股策略回测 - 改进版

改进点：
1. 动态买入阈值：Probability > Percentile(Historical_Probabilities, 80)
2. ATR止损止盈：止损 = N * ATR(14)
3. 正确的三阶段验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 数据获取
# ============================================================================

def fetch_real_data_baostock(stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """使用BaoStock获取真实数据"""
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
            if code.startswith('6'):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
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
            df = df.set_index('timestamp')
            df = df.sort_index()

            stock_data[code] = df
            print(f"  {code}: {len(df)} 条数据")

        except Exception:
            pass

    bs.logout()
    return stock_data


# ============================================================================
# 三阶段数据划分
# ============================================================================

def split_train_validation_test(
    stock_data: Dict[str, pd.DataFrame],
    train_start: str = "2020-01-01",
    train_end: str = "2024-03-31",
    validation_start: str = "2024-04-01",
    validation_end: str = "2024-12-31",  # 增加验证集到9个月
    test_start: str = "2025-01-01",
    test_end: str = "2026-02-28"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """三阶段数据划分"""
    train_data = {}
    validation_data = {}
    test_data = {}

    for code, df in stock_data.items():
        train_df = df.loc[train_start:train_end]
        validation_df = df.loc[validation_start:validation_end]
        test_df = df.loc[test_start:test_end]

        if len(train_df) > 100 and len(validation_df) > 50 and len(test_df) > 20:
            train_data[code] = train_df
            validation_data[code] = validation_df
            test_data[code] = test_df

    print(f"\n三阶段数据划分完成:")
    print(f"  训练集 ({train_start} ~ {train_end}): {len(train_data)} 只股票")
    print(f"  验证集 ({validation_start} ~ {validation_end}): {len(validation_data)} 只股票")
    print(f"  测试集 ({test_start} ~ {test_end}): {len(test_data)} 只股票")

    return train_data, validation_data, test_data


# ============================================================================
# ATR计算
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


# ============================================================================
# 改进的策略回测
# ============================================================================

def run_backtest_improved(
    train_data: Dict[str, pd.DataFrame],
    backtest_data: Dict[str, pd.DataFrame],
    position_range: Tuple[float, float] = (0.20, 0.50),
    probability_percentile: float = 80,  # 只买概率最高的 20%
    stop_loss_atr_mult: float = 2.0,  # 止损 = 2 * ATR
    take_profit_atr_mult: float = 3.0,  # 止盈 = 3 * ATR
    atr_period: int = 14,
    hold_days: int = 10,
    initial_capital: float = 1000000,
    max_positions: int = 5,
    # 模型超参数
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    feature_selection_threshold: int = 50
) -> Dict:
    """
    改进的回测引擎

    特点：
    1. 动态买入阈值：只买概率最高的 (100 - probability_percentile)% 的信号
    2. ATR止损止盈：根据市场波动自动调整
    """
    from quanttool.strategies.ml_stock_selection_strategy import MLStockSelectionStrategy

    if not train_data or not backtest_data:
        return {'error': '数据不足'}

    # 合并所有训练数据
    all_train_dfs = []
    for code, df in train_data.items():
        df_copy = df.copy()
        df_copy['code'] = code
        all_train_dfs.append(df_copy.reset_index())

    combined_train = pd.concat(all_train_dfs, ignore_index=True)

    # 初始化策略（买入阈值先用默认值，后面会动态调整）
    strategy = MLStockSelectionStrategy(
        buy_prob_threshold=0.5,  # 临时值，后面会动态计算
        sell_prob_threshold=0.3,
        position_range=position_range,
        stop_loss_pct=0.05,  # 临时值，后面用ATR
        take_profit_pct=0.10,
        hold_days=hold_days
    )

    # 训练模型
    model_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'feature_selection_threshold': feature_selection_threshold
    }
    train_success = strategy.train_model(combined_train, model_params=model_params)

    if not train_success:
        return {'error': '模型训练失败'}

    # 预先计算所有日期的概率和ATR
    all_probabilities = []
    date_code_probs = {}  # {date: {code: prob}}

    all_dates = set()
    for df in backtest_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(list(all_dates))

    # 第一遍：计算所有概率
    print("  预计算概率分布...")
    for date in all_dates:
        date_code_probs[date] = {}
        for code, df in backtest_data.items():
            if date not in df.index:
                continue

            hist_df = df.loc[:date].reset_index()
            if len(hist_df) < 120:
                continue

            try:
                features = strategy.feature_engineer.generate_features(hist_df)
                latest_features = features.iloc[[-1]]
                prob = strategy.trainer.predict_proba(latest_features)[0]

                date_code_probs[date][code] = prob
                all_probabilities.append(prob)
            except:
                pass

    # 计算概率阈值（动态：只买最高的20%）
    if all_probabilities:
        dynamic_threshold = np.percentile(all_probabilities, probability_percentile)
        print(f"  动态买入阈值 (P{probability_percentile}): {dynamic_threshold:.3f}")
    else:
        dynamic_threshold = 0.5
        print(f"  无法计算阈值，使用默认值: {dynamic_threshold:.3f}")

    # 第二遍：执行回测
    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [capital]

    signal_stats = {'buy': 0, 'hold': 0, 'total_checks': 0, 'low_prob': 0, 'bad_position': 0}

    for date in all_dates:
        daily_pnl = 0

        # 检查现有持仓
        for code in list(positions.keys()):
            if code not in backtest_data:
                continue
            df = backtest_data[code]
            if date not in df.index:
                continue

            close = df.loc[date, 'close']
            pos = positions[code]

            # ATR止损
            atr = calculate_atr(df.loc[:date], atr_period).iloc[-1]
            stop_loss_price = pos['entry_price'] - stop_loss_atr_mult * atr
            take_profit_price = pos['entry_price'] + take_profit_atr_mult * atr

            # 止损
            if close <= stop_loss_price:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'stop_loss', 'date': date})
                del positions[code]

            # 止盈
            elif close >= take_profit_price:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'take_profit', 'date': date})
                del positions[code]

            # 超时
            elif (date - pos['entry_date']).days >= hold_days * 2:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'timeout', 'date': date})
                del positions[code]

        # 尝试新买入
        if len(positions) < max_positions:
            for code, df in backtest_data.items():
                if code in positions:
                    continue
                if date not in df.index:
                    continue

                hist_df = df.loc[:date].reset_index()
                if len(hist_df) < 120:
                    continue

                try:
                    signal = strategy.get_signal(hist_df.iloc[-1], hist_df)
                    signal_stats['total_checks'] += 1

                    # 使用动态阈值
                    prob = date_code_probs.get(date, {}).get(code, 0)
                    # 修复: get_signal 返回的是 'position'，不是 'position_score'
                    position = signal.get('position', 0)

                    # 买入条件：概率高于动态阈值 + 位置在范围内
                    if prob >= dynamic_threshold and position_range[0] <= position <= position_range[1]:
                        signal_stats['buy'] += 1
                        close = df.loc[date, 'close']
                        atr = calculate_atr(df.loc[:date], atr_period).iloc[-1]

                        position_value = capital * 0.18
                        shares = position_value / close

                        positions[code] = {
                            'shares': shares,
                            'entry_price': close,
                            'stop_loss': close - stop_loss_atr_mult * atr,
                            'take_profit': close + take_profit_atr_mult * atr,
                            'entry_date': date,
                            'atr': atr
                        }

                        trades.append({'code': code, 'action': 'buy', 'price': close, 'shares': shares, 'date': date, 'prob': prob, 'atr': atr})
                    else:
                        signal_stats['hold'] += 1
                        if prob < dynamic_threshold:
                            signal_stats['low_prob'] += 1
                        else:
                            signal_stats['bad_position'] += 1

                except Exception:
                    pass

        capital += daily_pnl
        equity_curve.append(capital)

    # 打印信号统计
    print(f"\n  信号统计: 检查 {signal_stats['total_checks']} 次, 买入 {signal_stats['buy']} 次, 持有 {signal_stats['hold']} 次")
    if signal_stats['hold'] > 0:
        print(f"    低概率: {signal_stats['low_prob']}, 位置不佳: {signal_stats['bad_position']}")

    # 平剩余仓位
    for code, pos in list(positions.items()):
        if code in backtest_data:
            df = backtest_data[code]
            if len(df) > 0:
                close = df.iloc[-1]['close']
                pnl = (close - pos['entry_price']) * pos['shares']
                capital += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'final', 'date': df.index[-1]})

    # 计算指标
    sell_trades = [t for t in trades if t.get('action') == 'sell']
    winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing = [t for t in sell_trades if t.get('pnl', 0) <= 0]

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
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(sell_trades) if sell_trades else 0,
        'trades': trades,
        'equity_curve': equity_curve,
        'dynamic_threshold': dynamic_threshold,
        'params': {
            'position_range': position_range,
            'probability_percentile': probability_percentile,
            'stop_loss_atr_mult': stop_loss_atr_mult,
            'take_profit_atr_mult': take_profit_atr_mult,
            'atr_period': atr_period
        }
    }


# ============================================================================
# 参数稳健性检验
# ============================================================================

def check_robustness(
    train_data: Dict[str, pd.DataFrame],
    validation_data: Dict[str, pd.DataFrame],
    base_params: Dict,
    perturbation_ratio: float = 0.10
) -> Tuple[bool, Dict]:
    """检查参数稳健性"""
    results = []

    base_result = run_backtest_improved(
        train_data=train_data,
        backtest_data=validation_data,
        **base_params
    )

    if 'error' in base_result:
        return False, {'error': base_result['error']}

    results.append(base_result['annual_return'])

    # 生成扰动参数组合
    perturbations = []

    # 扰动 probability_percentile
    for delta in [-5, 5]:
        new_pct = base_params['probability_percentile'] + delta
        new_pct = max(50, min(95, new_pct))
        params = base_params.copy()
        params['probability_percentile'] = new_pct
        perturbations.append(params)

    # 扰动 stop_loss_atr_mult
    for delta in [-0.3, 0.3]:
        new_mult = base_params['stop_loss_atr_mult'] + delta
        new_mult = max(1.0, min(4.0, new_mult))
        params = base_params.copy()
        params['stop_loss_atr_mult'] = new_mult
        perturbations.append(params)

    # 扰动 take_profit_atr_mult
    for delta in [-0.5, 0.5]:
        new_mult = base_params['take_profit_atr_mult'] + delta
        new_mult = max(1.5, min(5.0, new_mult))
        params = base_params.copy()
        params['take_profit_atr_mult'] = new_mult
        perturbations.append(params)

    for params in perturbations:
        try:
            result = run_backtest_improved(
                train_data=train_data,
                backtest_data=validation_data,
                **params
            )
            if 'error' not in result:
                results.append(result['annual_return'])
        except:
            pass

    if len(results) < 3:
        return False, {
            'base_return': base_result['annual_return'],
            'error': '扰动测试数据不足'
        }

    mean_return = np.mean(results)
    std_return = np.std(results)
    cv = std_return / abs(mean_return) if mean_return != 0 else float('inf')

    is_robust = cv < 0.5

    return is_robust, {
        'base_return': base_result['annual_return'],
        'mean_return': mean_return,
        'std_return': std_return,
        'coefficient_of_variation': cv,
        'is_robust': is_robust
    }


# ============================================================================
# 验证集参数优化
# ============================================================================

def optimize_on_validation(
    train_data: Dict[str, pd.DataFrame],
    validation_data: Dict[str, pd.DataFrame],
    target_return: float = 0.15,
    max_iterations: int = 50,
    require_robustness: bool = True
) -> Tuple[Dict, Dict]:
    """在验证集上优化参数 - 增加正则化防止过拟合"""

    import random

    # 参数范围 - 第3轮：更保守的参数，减少过拟合
    position_min_range = [0.10, 0.15]  # 固定较小范围
    position_max_range = [0.85, 0.90]  # 固定较小范围
    probability_percentile_range = [55, 60, 65]  # 提高阈值，减少交易
    stop_loss_atr_range = [2.0, 2.5]  # 固定止损
    take_profit_atr_range = [3.0, 3.5]  # 固定止盈

    # 模型超参数 - 极简模型，强正则化
    n_estimators_range = [100, 150]  # 更少的树
    max_depth_range = [2, 3]  # 更浅的树
    learning_rate_range = [0.01]  # 更低学习率
    feature_selection_range = [15, 20]  # 更少的特征

    # 生成有效参数组合
    all_combos = []
    for pmin in position_min_range:
        for pmax in position_max_range:
            if pmax > pmin:
                for pct in probability_percentile_range:
                    for sl_atr in stop_loss_atr_range:
                        for tp_atr in take_profit_atr_range:
                            if tp_atr > sl_atr:  # 止盈 > 止损
                                for n_est in n_estimators_range:
                                    for md in max_depth_range:
                                        for lr in learning_rate_range:
                                            for fs in feature_selection_range:
                                                all_combos.append({
                                                    'position_range': (pmin, pmax),
                                                    'probability_percentile': pct,
                                                    'stop_loss_atr_mult': sl_atr,
                                                    'take_profit_atr_mult': tp_atr,
                                                    'n_estimators': n_est,
                                                    'max_depth': md,
                                                    'learning_rate': lr,
                                                    'feature_selection_threshold': fs
                                                })

    print(f"总参数组合数: {len(all_combos)}")

    if len(all_combos) > max_iterations:
        param_sets = random.sample(all_combos, max_iterations)
    else:
        param_sets = all_combos

    best_params = None
    best_validation_return = -1
    optimization_log = []

    for i, params in enumerate(param_sets):
        print(f"\n{'='*50}")
        print(f"验证集优化 {i+1}/{len(param_sets)}")
        print(f"参数: {params}")

        try:
            result = run_backtest_improved(
                train_data=train_data,
                backtest_data=validation_data,
                **params
            )

            if 'error' in result:
                print(f"  错误: {result['error']}")
                continue

            annual = result['annual_return']
            print(f"  验证集年化收益: {annual:.2%}")

            optimization_log.append({
                'iteration': i + 1,
                'params': params,
                'validation_return': annual
            })

            # 检查稳健性
            if require_robustness and annual > target_return:
                is_robust, robustness_report = check_robustness(
                    train_data=train_data,
                    validation_data=validation_data,
                    base_params=params
                )

                if not is_robust:
                    print(f"  稳健性检验未通过 (CV={robustness_report.get('coefficient_of_variation', 0):.2f})")
                    continue

                print(f"  ✓ 稳健性检验通过")

            if annual > best_validation_return:
                best_validation_return = annual
                best_params = params

            if annual >= target_return:
                print(f"\n✓ 达到目标年化收益 {target_return:.0%}!")
                return params, {
                    'validation_return': annual,
                    'optimization_log': optimization_log[:i+1],
                    'found_target': True
                }

        except Exception as e:
            print(f"  异常: {e}")
            continue

    print(f"\n验证集最优年化收益: {best_validation_return:.2%}")

    return best_params, {
        'validation_return': best_validation_return,
        'optimization_log': optimization_log,
        'found_target': False
    }


# ============================================================================
# 最终测试集评估
# ============================================================================

def final_test_evaluation(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    locked_params: Dict
) -> Dict:
    """最终测试集评估 - 只运行一次"""
    print("\n" + "=" * 60)
    print("⚠️  最终测试集评估 - 只运行一次")
    print("=" * 60)
    print(f"使用锁定参数: {locked_params}")

    result = run_backtest_improved(
        train_data=train_data,
        backtest_data=test_data,
        **locked_params
    )

    return result


# ============================================================================
# 过拟合检测
# ============================================================================

def detect_overfitting(
    validation_return: float,
    test_return: float,
    threshold: float = 0.30
) -> Tuple[bool, Dict]:
    """检测过拟合"""
    if validation_return <= 0:
        return True, {
            'is_overfitting': True,
            'reason': '验证集收益为负'
        }

    drop_ratio = (validation_return - test_return) / validation_return
    is_overfitting = drop_ratio > threshold

    return is_overfitting, {
        'is_overfitting': is_overfitting,
        'validation_return': validation_return,
        'test_return': test_return,
        'drop_ratio': drop_ratio,
        'threshold': threshold,
        'reason': f"验证集 -> 测试集收益下降 {drop_ratio:.1%}" if is_overfitting else "收益下降在可接受范围内"
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    """主函数 - 改进的三阶段验证流程"""
    stock_codes = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

    print("=" * 60)
    print("机器学习选股策略 - 改进版")
    print("=" * 60)
    print("""
改进点：
1. 动态买入阈值：只买概率最高的 20% 信号 (自适应)
2. ATR止损止盈：根据市场波动自动调整

三阶段数据划分:
  训练集 (2020-01-01 ~ 2024-06-30): 训练模型
  验证集 (2024-07-01 ~ 2024-12-31): 参数优化
  测试集 (2025-01-01 ~ 2026-02-28): 最终评估
""")

    # 1. 获取数据
    print("\n" + "=" * 60)
    print("Step 1: 获取真实数据")
    print("=" * 60)
    stock_data = fetch_real_data_baostock(
        stock_codes=stock_codes,
        start_date="2020-01-01",
        end_date="2026-02-28"
    )

    if not stock_data:
        print("获取数据失败")
        return

    # 2. 三阶段数据划分
    print("\n" + "=" * 60)
    print("Step 2: 三阶段数据划分")
    print("=" * 60)
    train_data, validation_data, test_data = split_train_validation_test(stock_data)

    if not train_data or not validation_data or not test_data:
        print("数据划分失败")
        return

    # 3. 在验证集上优化参数
    print("\n" + "=" * 60)
    print("Step 3: 在验证集上优化参数")
    print("=" * 60)
    best_params, optimization_report = optimize_on_validation(
        train_data=train_data,
        validation_data=validation_data,
        target_return=0.15,
        max_iterations=50,
        require_robustness=True
    )

    if not best_params:
        print("参数优化失败")
        return

    print(f"\n验证集最优参数: {best_params}")
    print(f"验证集年化收益: {optimization_report['validation_return']:.2%}")

    # 4. 最终测试集评估 (只运行一次)
    print("\n" + "=" * 60)
    print("Step 4: 最终测试集评估")
    print("=" * 60)
    final_result = final_test_evaluation(
        train_data=train_data,
        test_data=test_data,
        locked_params=best_params
    )

    if 'error' in final_result:
        print(f"测试失败: {final_result['error']}")
        return

    # 5. 过拟合检测
    print("\n" + "=" * 60)
    print("Step 5: 过拟合检测")
    print("=" * 60)
    is_overfitting, overfitting_report = detect_overfitting(
        validation_return=optimization_report['validation_return'],
        test_return=final_result['annual_return'],
        threshold=0.30
    )

    # 6. 打印最终结果
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)

    print(f"\n{'='*30}")
    print("锁定参数")
    print(f"{'='*30}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*30}")
    print("验证集表现")
    print(f"{'='*30}")
    print(f"  年化收益: {optimization_report['validation_return']:.2%}")

    print(f"\n{'='*30}")
    print("测试集表现 (最终)")
    print(f"{'='*30}")
    print(f"  初始资金: {final_result['initial_capital']:,.0f}")
    print(f"  最终资金: {final_result['final_capital']:,.0f}")
    print(f"  总收益率: {final_result['total_return']:.2%}")
    print(f"  年化收益: {final_result['annual_return']:.2%}")
    print(f"  最大回撤: {final_result['max_drawdown']:.2%}")
    print(f"  总交易次数: {final_result['total_trades']}")
    print(f"  胜率: {final_result['win_rate']:.2%}")
    print(f"  动态阈值: {final_result.get('dynamic_threshold', 0):.3f}")

    print(f"\n{'='*30}")
    print("过拟合检测")
    print(f"{'='*30}")
    if is_overfitting:
        print(f"  ⚠️  检测到过拟合!")
        print(f"  验证集 -> 测试集收益下降: {overfitting_report['drop_ratio']:.1%}")
        print(f"  阈值: {overfitting_report['threshold']:.0%}")
    else:
        print(f"  ✓ 未检测到过拟合")
        print(f"  验证集 -> 测试集收益下降: {overfitting_report.get('drop_ratio', 0):.1%}")

    print(f"\n{'='*30}")
    print("结论")
    print(f"{'='*30}")
    if final_result['annual_return'] >= 0.15 and not is_overfitting:
        print(f"  ✓ 策略验证通过，年化收益 {final_result['annual_return']:.2%} >= 15%")
    elif is_overfitting:
        print(f"  ✗ 策略存在过拟合风险，需要重新设计")
    else:
        print(f"  ✗ 年化收益 {final_result['annual_return']:.2%} 未达到目标 15%")

    return {
        'best_params': best_params,
        'validation_return': optimization_report['validation_return'],
        'test_return': final_result['annual_return'],
        'is_overfitting': is_overfitting,
        'final_result': final_result
    }


if __name__ == "__main__":
    main()