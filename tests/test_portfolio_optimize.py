#!/usr/bin/env python
"""
趋势动量策略组合优化

目标: 年化收益 > 15%

两阶段优化:
1. 网格搜索外层参数 (buy_threshold, sell_threshold, stop_loss_pct, take_profit_pct)
2. 贝叶斯优化内层评分参数
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import asdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from itertools import product

from quanttool.factors.trend_momentum_scoring import ScoringConfig, TrendMomentumScoring
from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy
from quanttool.backtest.engine import BacktestEngine

# 尝试导入数据源
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


# 目标股票
TARGET_STOCKS = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

# 回测参数
INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365
TARGET_RETURN = 0.15  # 15%

# 网格搜索参数 - 精简版
OUTER_PARAMS = {
    'buy_threshold': [50, 55, 60, 65],
    'sell_threshold': [30, 35, 40],
    'stop_loss_pct': [0.05, 0.07, 0.10],
    'take_profit_pct': [0.10, 0.15, 0.20],
}

# 贝叶斯优化参数范围
INNER_PARAMS_RANGE = {
    'mom_5_strong': (2.0, 5.0),
    'mom_10_strong': (3.0, 8.0),
    'mom_20_strong': (6.0, 15.0),
    'ma20_slope_threshold': (1.0, 4.0),
    'vol_ratio_huge': (1.5, 3.0),
    'position_mid_low': (0.2, 0.4),
    'position_mid_high': (0.5, 0.7),
}


def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """获取股票历史数据 - 使用BaoStock"""
    if not BAOSTOCK_AVAILABLE:
        print("BaoStock 未安装")
        return pd.DataFrame()

    try:
        # 登录 baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f"BaoStock 登录失败: {lg.error_msg}")
            return pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS + 100)

        # 转换代码格式
        bs_code = f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            frequency="d",
            adjustflag="2"  # 前复权
        )

        if rs.error_code != '0':
            return pd.DataFrame()

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        df['timestamp'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna().sort_values('timestamp').reset_index(drop=True)

        if len(df) > LOOKBACK_DAYS:
            df = df.tail(LOOKBACK_DAYS).reset_index(drop=True)

        return df

    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return pd.DataFrame()


class PortfolioBacktester:
    """组合回测器"""

    def __init__(self, stocks: List[str], initial_capital: float = INITIAL_CAPITAL):
        self.stocks = stocks
        self.initial_capital = initial_capital
        self.stock_data: Dict[str, pd.DataFrame] = {}

    def load_data(self) -> int:
        """加载所有股票数据"""
        loaded = 0
        for symbol in self.stocks:
            df = fetch_stock_data(symbol)
            if not df.empty and len(df) >= 60:
                self.stock_data[symbol] = df
                loaded += 1
                print(f"  ✅ {symbol}: {len(df)} 条数据")
            else:
                print(f"  ⚠️ {symbol}: 数据不足")
        return loaded

    def run_backtest(
        self,
        config: ScoringConfig,
    ) -> Dict[str, Any]:
        """
        运行组合回测

        Args:
            config: 评分配置

        Returns:
            回测结果字典
        """
        # 创建评分系统
        scoring = TrendMomentumScoring(config=config)

        # 创建策略
        strategy = TrendMomentumStrategy(
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
        )

        # 更新策略的评分系统
        strategy.scoring_system = scoring

        # 运行回测
        engine = BacktestEngine()
        engine.set_initial_cash(self.initial_capital)

        try:
            # 获取统一的开始和结束日期
            all_timestamps = set()
            for symbol, df in self.stock_data.items():
                all_timestamps.update(df['timestamp'].tolist())
            all_timestamps = sorted(list(all_timestamps))

            start_date = all_timestamps[0]
            end_date = all_timestamps[-1]

            result = engine.run_backtest(
                strategy=strategy,
                data=self.stock_data,
                start_date=start_date,
                end_date=end_date
            )

            return {
                'success': True,
                'annual_return': result.annual_return,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'annual_return': -1.0}

    def calculate_objective(self, result: Dict[str, Any]) -> float:
        """
        计算目标函数值

        objective = annual_return - 0.5 * max_drawdown + 0.1 * sharpe_ratio
        """
        if not result.get('success', False):
            return -float('inf')

        annual_return = result.get('annual_return', 0)
        max_drawdown = result.get('max_drawdown', 0)
        sharpe_ratio = result.get('sharpe_ratio', 0)

        objective = annual_return - 0.5 * max_drawdown + 0.1 * sharpe_ratio
        return objective


class PortfolioOptimizer:
    """组合参数优化器"""

    def __init__(self, backtester: PortfolioBacktester):
        self.backtester = backtester
        self.best_config: Optional[ScoringConfig] = None
        self.best_result: Optional[Dict] = None
        self.best_objective: float = -float('inf')
        self.optimization_history: List[Dict] = []

    def grid_search_outer_params(self) -> ScoringConfig:
        """
        阶段1: 网格搜索外层参数

        Returns:
            最佳配置
        """
        print("\n" + "=" * 70)
        print("阶段1: 网格搜索外层参数")
        print("=" * 70)

        # 生成所有参数组合
        param_names = list(OUTER_PARAMS.keys())
        param_values = list(OUTER_PARAMS.values())
        all_combinations = list(product(*param_values))

        print(f"总参数组合数: {len(all_combinations)}")

        tested = 0
        for combo in all_combinations:
            params = dict(zip(param_names, combo))

            # 买入阈值必须大于卖出阈值
            if params['buy_threshold'] <= params['sell_threshold']:
                continue

            # 创建配置
            config = ScoringConfig(**params)

            # 运行回测
            result = self.backtester.run_backtest(config)
            tested += 1

            # 计算目标函数
            objective = self.backtester.calculate_objective(result)

            # 记录历史
            self.optimization_history.append({
                'phase': 'grid_search',
                'params': params,
                'result': result,
                'objective': objective,
            })

            # 更新最佳
            if objective > self.best_objective:
                self.best_objective = objective
                self.best_config = config
                self.best_result = result

                print(f"  [{tested}] 新最佳: 年化={result['annual_return']*100:.2f}%, "
                      f"回撤={result['max_drawdown']*100:.2f}%, 目标={objective:.4f}")

                # 如果达到目标，提前退出
                if result['annual_return'] >= TARGET_RETURN:
                    print(f"\n  ✅ 达到目标年化收益 {TARGET_RETURN*100}%!")
                    return config

        print(f"\n网格搜索完成，测试 {tested} 组参数")
        print(f"最佳年化收益: {self.best_result['annual_return']*100:.2f}%")

        return self.best_config

    def bayesian_optimize_inner_params(
        self,
        base_config: ScoringConfig,
        n_iterations: int = 50
    ) -> ScoringConfig:
        """
        阶段2: 贝叶斯优化内层参数

        使用简化的贝叶斯优化（随机搜索 + 适应性采样）

        Args:
            base_config: 基础配置
            n_iterations: 迭代次数

        Returns:
            最佳配置
        """
        print("\n" + "=" * 70)
        print("阶段2: 贝叶斯优化内层参数")
        print("=" * 70)

        # 尝试使用 scikit-optimize
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            use_skopt = True
            print("使用 scikit-optimize 进行贝叶斯优化")
        except ImportError:
            use_skopt = False
            print("scikit-optimize 未安装，使用随机搜索")

        if use_skopt:
            return self._skopt_optimize(base_config, n_iterations)
        else:
            return self._random_search_optimize(base_config, n_iterations)

    def _skopt_optimize(self, base_config: ScoringConfig, n_iterations: int) -> ScoringConfig:
        """使用 scikit-optimize 进行贝叶斯优化"""
        from skopt import gp_minimize
        from skopt.space import Real

        # 定义参数空间
        space = [
            Real(INNER_PARAMS_RANGE['mom_5_strong'][0], INNER_PARAMS_RANGE['mom_5_strong'][1], name='mom_5_strong'),
            Real(INNER_PARAMS_RANGE['mom_10_strong'][0], INNER_PARAMS_RANGE['mom_10_strong'][1], name='mom_10_strong'),
            Real(INNER_PARAMS_RANGE['mom_20_strong'][0], INNER_PARAMS_RANGE['mom_20_strong'][1], name='mom_20_strong'),
            Real(INNER_PARAMS_RANGE['ma20_slope_threshold'][0], INNER_PARAMS_RANGE['ma20_slope_threshold'][1], name='ma20_slope_threshold'),
            Real(INNER_PARAMS_RANGE['vol_ratio_huge'][0], INNER_PARAMS_RANGE['vol_ratio_huge'][1], name='vol_ratio_huge'),
            Real(INNER_PARAMS_RANGE['position_mid_low'][0], INNER_PARAMS_RANGE['position_mid_low'][1], name='position_mid_low'),
            Real(INNER_PARAMS_RANGE['position_mid_high'][0], INNER_PARAMS_RANGE['position_mid_high'][1], name='position_mid_high'),
        ]

        param_names = ['mom_5_strong', 'mom_10_strong', 'mom_20_strong',
                       'ma20_slope_threshold', 'vol_ratio_huge',
                       'position_mid_low', 'position_mid_high']

        def objective(params_list):
            params = dict(zip(param_names, params_list))

            # 创建配置
            config = ScoringConfig(
                buy_threshold=base_config.buy_threshold,
                sell_threshold=base_config.sell_threshold,
                stop_loss_pct=base_config.stop_loss_pct,
                take_profit_pct=base_config.take_profit_pct,
                **params
            )

            # 运行回测
            result = self.backtester.run_backtest(config)

            # 返回负的目标函数（因为 skopt 是最小化）
            objective_value = self.backtester.calculate_objective(result)

            # 记录历史
            self.optimization_history.append({
                'phase': 'bayesian',
                'params': {**{'buy_threshold': base_config.buy_threshold,
                             'sell_threshold': base_config.sell_threshold,
                             'stop_loss_pct': base_config.stop_loss_pct,
                             'take_profit_pct': base_config.take_profit_pct}, **params},
                'result': result,
                'objective': objective_value,
            })

            # 更新最佳
            if objective_value > self.best_objective:
                self.best_objective = objective_value
                self.best_config = config
                self.best_result = result
                print(f"  新最佳: 年化={result['annual_return']*100:.2f}%, 目标={objective_value:.4f}")

            return -objective_value

        # 运行优化
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=42,
            verbose=False
        )

        print(f"\n贝叶斯优化完成，总迭代 {n_iterations} 次")
        print(f"最佳年化收益: {self.best_result['annual_return']*100:.2f}%")

        return self.best_config

    def _random_search_optimize(self, base_config: ScoringConfig, n_iterations: int) -> ScoringConfig:
        """随机搜索优化"""
        print(f"随机搜索 {n_iterations} 次迭代...")

        for i in range(n_iterations):
            # 随机采样参数
            params = {}
            for name, (low, high) in INNER_PARAMS_RANGE.items():
                params[name] = np.random.uniform(low, high)

            # 确保 position_mid_low < position_mid_high
            if params['position_mid_low'] >= params['position_mid_high']:
                params['position_mid_low'], params['position_mid_high'] = \
                    min(params['position_mid_low'], params['position_mid_high']), \
                    max(params['position_mid_low'], params['position_mid_high']) + 0.1

            # 创建配置
            config = ScoringConfig(
                buy_threshold=base_config.buy_threshold,
                sell_threshold=base_config.sell_threshold,
                stop_loss_pct=base_config.stop_loss_pct,
                take_profit_pct=base_config.take_profit_pct,
                **params
            )

            # 运行回测
            result = self.backtester.run_backtest(config)

            # 计算目标函数
            objective_value = self.backtester.calculate_objective(result)

            # 记录历史
            self.optimization_history.append({
                'phase': 'random_search',
                'params': {**{'buy_threshold': base_config.buy_threshold,
                             'sell_threshold': base_config.sell_threshold,
                             'stop_loss_pct': base_config.stop_loss_pct,
                             'take_profit_pct': base_config.take_profit_pct}, **params},
                'result': result,
                'objective': objective_value,
            })

            # 更新最佳
            if objective_value > self.best_objective:
                self.best_objective = objective_value
                self.best_config = config
                self.best_result = result
                print(f"  [{i+1}] 新最佳: 年化={result['annual_return']*100:.2f}%, "
                      f"回撤={result['max_drawdown']*100:.2f}%, 目标={objective_value:.4f}")

            # 如果达到目标，提前退出
            if result['annual_return'] >= TARGET_RETURN:
                print(f"\n  ✅ 达到目标年化收益 {TARGET_RETURN*100}%!")
                break

        print(f"\n随机搜索完成，迭代 {i+1} 次")
        print(f"最佳年化收益: {self.best_result['annual_return']*100:.2f}%")

        return self.best_config

    def save_results(self, output_path: str):
        """保存优化结果"""
        # 转换为可序列化格式
        output = {
            'timestamp': datetime.now().isoformat(),
            'target_return': TARGET_RETURN,
            'stocks': TARGET_STOCKS,
            'best_config': self.best_config.to_dict() if self.best_config else None,
            'best_result': self.best_result,
            'best_objective': self.best_objective,
            'reached_target': self.best_result and self.best_result.get('annual_return', 0) >= TARGET_RETURN,
            'optimization_history': [
                {
                    'phase': h['phase'],
                    'params': h['params'],
                    'objective': h['objective'],
                    'annual_return': h['result'].get('annual_return', 0),
                    'max_drawdown': h['result'].get('max_drawdown', 0),
                    'sharpe_ratio': h['result'].get('sharpe_ratio', 0),
                }
                for h in self.optimization_history
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n优化结果已保存到: {output_path}")


def main():
    print("=" * 70)
    print("趋势动量策略组合优化 - 目标年化收益 > 15%")
    print("=" * 70)
    print(f"目标股票: {TARGET_STOCKS}")
    print(f"回测期间: 最近 {LOOKBACK_DAYS} 天")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")

    # 创建回测器
    backtester = PortfolioBacktester(TARGET_STOCKS, INITIAL_CAPITAL)

    # 加载数据
    print("\n加载股票数据...")
    loaded = backtester.load_data()

    if loaded == 0:
        print("\n❌ 没有有效数据，退出")
        return

    # 创建优化器
    optimizer = PortfolioOptimizer(backtester)

    # 阶段1: 网格搜索外层参数
    best_outer_config = optimizer.grid_search_outer_params()

    # 阶段2: 贝叶斯优化内层参数
    best_config = optimizer.bayesian_optimize_inner_params(best_outer_config, n_iterations=50)

    # 输出最终结果
    print("\n" + "=" * 70)
    print("📊 最终优化结果")
    print("=" * 70)

    if optimizer.best_result:
        result = optimizer.best_result
        config = optimizer.best_config

        print(f"\n年化收益: {result['annual_return']*100:.2f}%")
        print(f"总收益: {result['total_return']*100:.2f}%")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"胜率: {result['win_rate']*100:.1f}%")
        print(f"交易次数: {result['total_trades']}")

        print(f"\n最佳参数:")
        print(f"  买入阈值: {config.buy_threshold}")
        print(f"  卖出阈值: {config.sell_threshold}")
        print(f"  止损比例: {config.stop_loss_pct*100}%")
        print(f"  止盈比例: {config.take_profit_pct*100}%")
        print(f"  5日动量强阈值: {config.mom_5_strong:.2f}")
        print(f"  10日动量强阈值: {config.mom_10_strong:.2f}")
        print(f"  20日动量强阈值: {config.mom_20_strong:.2f}")
        print(f"  MA20斜率阈值: {config.ma20_slope_threshold:.2f}")
        print(f"  巨量量比阈值: {config.vol_ratio_huge:.2f}")
        print(f"  位置区间: [{config.position_mid_low:.2f}, {config.position_mid_high:.2f}]")

        # 验证目标
        if result['annual_return'] >= TARGET_RETURN:
            print(f"\n✅ 成功达到目标年化收益 {TARGET_RETURN*100}%!")
        else:
            print(f"\n⚠️ 未达到目标年化收益 {TARGET_RETURN*100}%")
            print("建议: 可能需要调整股票池或策略逻辑")

        # 保存结果
        output_path = f"reports/portfolio_optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.save_results(output_path)

    return optimizer


if __name__ == "__main__":
    optimizer = main()