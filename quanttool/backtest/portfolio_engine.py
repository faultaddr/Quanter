"""
组合回测引擎模块

实现多股票组合回测：
- 多股票评分计算
- 组合构建与再平衡
- 风险平价/等权分配
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class RebalanceFrequency(str, Enum):
    """再平衡频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class WeightMethod(str, Enum):
    """权重方法"""
    EQUAL = "equal"               # 等权
    RISK_PARITY = "risk_parity"   # 风险平价
    SCORE_WEIGHTED = "score"      # 评分加权
    VOLATILITY_TARGET = "vol_target"  # 波动率目标


@dataclass
class PortfolioPosition:
    """组合持仓"""
    stock_code: str
    shares: float
    entry_price: float
    current_price: float
    market_value: float
    weight: float
    entry_date: datetime
    days_held: int


@dataclass
class PortfolioSnapshot:
    """组合快照"""
    date: datetime
    total_value: float
    cash: float
    positions: List[PortfolioPosition]
    returns: float
    drawdown: float


@dataclass
class PortfolioBacktestResult:
    """组合回测结果"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    snapshots: List[PortfolioSnapshot]
    trade_history: List[Dict]


class PortfolioBacktestEngine:
    """
    组合回测引擎

    执行多股票组合的回测
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        max_positions: int = 10,
        position_size: float = 0.1,
        weight_method: WeightMethod = WeightMethod.EQUAL,
        rebalance_freq: RebalanceFrequency = RebalanceFrequency.WEEKLY,
        buy_threshold: float = 70.0,
        sell_threshold: float = 50.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.001
    ):
        """
        初始化组合回测引擎

        Args:
            initial_capital: 初始资金
            max_positions: 最大持仓数
            position_size: 单只股票仓位比例
            weight_method: 权重方法
            rebalance_freq: 再平衡频率
            buy_threshold: 买入评分阈值
            sell_threshold: 卖出评分阈值
            commission_rate: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size = position_size
        self.weight_method = weight_method
        self.rebalance_freq = rebalance_freq
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.commission_rate = commission_rate
        self.slippage = slippage

        # 状态
        self.cash = initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.snapshots: List[PortfolioSnapshot] = []
        self.trade_history: List[Dict] = []

        # 历史数据
        self.peak_value = initial_capital

    def run_portfolio_backtest(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        score_calculator: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PortfolioBacktestResult:
        """
        执行组合回测

        Args:
            stock_data_dict: {stock_code: df} 数据字典
            score_calculator: 评分计算函数
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            PortfolioBacktestResult: 回测结果
        """
        # 获取所有交易日
        all_dates = self._get_all_trading_dates(stock_data_dict)

        if not all_dates:
            raise ValueError("无法获取交易日历")

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        # 初始化
        self.cash = self.initial_capital
        self.positions = {}
        self.snapshots = []
        self.trade_history = []
        self.peak_value = self.initial_capital

        # 上次再平衡日期
        last_rebalance = None

        # 遍历交易日
        for i, date in enumerate(all_dates):
            # 检查是否需要再平衡
            need_rebalance = self._should_rebalance(date, last_rebalance)

            if need_rebalance:
                # 计算所有股票评分
                scores = self._calculate_scores(
                    date, stock_data_dict, score_calculator
                )

                # 执行再平衡
                self._rebalance(date, scores, stock_data_dict)

                last_rebalance = date

            # 更新持仓市值
            self._update_positions(date, stock_data_dict)

            # 记录快照
            self._record_snapshot(date)

        # 计算结果
        return self._calculate_result(all_dates[0], all_dates[-1])

    def _get_all_trading_dates(
        self,
        stock_data_dict: Dict[str, pd.DataFrame]
    ) -> List[datetime]:
        """获取所有交易日"""
        all_dates = set()

        for stock_code, df in stock_data_dict.items():
            if 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp']).dt.date.tolist()
            elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                dates = df.index.date.tolist()
            else:
                continue

            all_dates.update(dates)

        return sorted(list(all_dates))

    def _should_rebalance(
        self,
        current_date: datetime,
        last_rebalance: Optional[datetime]
    ) -> bool:
        """判断是否需要再平衡"""
        if last_rebalance is None:
            return True

        if self.rebalance_freq == RebalanceFrequency.DAILY:
            return True
        elif self.rebalance_freq == RebalanceFrequency.WEEKLY:
            # 周一或距离上次超过7天
            if current_date.weekday() == 0:
                return True
            return (current_date - last_rebalance).days >= 7
        elif self.rebalance_freq == RebalanceFrequency.MONTHLY:
            # 月初或距离上次超过30天
            if current_date.day <= 5:
                return True
            return (current_date - last_rebalance).days >= 30

        return False

    def _calculate_scores(
        self,
        date: datetime,
        stock_data_dict: Dict[str, pd.DataFrame],
        score_calculator: Callable
    ) -> Dict[str, float]:
        """计算所有股票评分"""
        scores = {}

        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)

        for stock_code, df in stock_data_dict.items():
            try:
                # 获取截止到当前日期的数据
                if 'timestamp' in df.columns:
                    mask = pd.to_datetime(df['timestamp']).dt.date <= date
                    historical_df = df[mask]
                else:
                    historical_df = df[df.index.date <= date]

                if len(historical_df) < 30:
                    continue

                # 计算评分
                score_result = score_calculator(historical_df)
                score = score_result.get('final_score', 50)
                scores[stock_code] = score

            except Exception as e:
                continue

        return scores

    def _rebalance(
        self,
        date: datetime,
        scores: Dict[str, float],
        stock_data_dict: Dict[str, pd.DataFrame]
    ):
        """执行再平衡"""
        # 获取当前价格
        prices = {}
        for stock_code in scores.keys():
            if stock_code in stock_data_dict:
                df = stock_data_dict[stock_code]
                if 'timestamp' in df.columns:
                    mask = pd.to_datetime(df['timestamp']).dt.date <= date
                    price_df = df[mask]
                else:
                    price_df = df[df.index.date <= date]

                if not price_df.empty:
                    prices[stock_code] = price_df['close'].iloc[-1]

        # 卖出信号
        for stock_code, position in list(self.positions.items()):
            score = scores.get(stock_code, 50)

            if score < self.sell_threshold:
                # 卖出
                self._sell_position(stock_code, prices.get(stock_code, 0), date)

        # 买入信号
        # 筛选高分股票
        buy_candidates = [
            (code, score) for code, score in scores.items()
            if score >= self.buy_threshold and code not in self.positions
        ]

        # 排序
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        # 计算可用仓位
        available_slots = self.max_positions - len(self.positions)

        for stock_code, score in buy_candidates[:available_slots]:
            price = prices.get(stock_code)
            if price and price > 0:
                self._buy_position(stock_code, price, score, date)

        # 调整权重（如果使用风险平价）
        if self.weight_method == WeightMethod.RISK_PARITY:
            self._adjust_risk_parity_weights(date, stock_data_dict)

    def _buy_position(
        self,
        stock_code: str,
        price: float,
        score: float,
        date: datetime
    ):
        """买入持仓"""
        # 考虑滑点
        buy_price = price * (1 + self.slippage)

        # 计算买入金额
        position_value = self.initial_capital * self.position_size

        # 检查资金是否足够
        if position_value > self.cash:
            position_value = self.cash * 0.95  # 保留5%现金

        # 计算股数
        shares = position_value / buy_price
        shares = int(shares / 100) * 100  # 整手

        if shares <= 0:
            return

        # 计算实际金额和手续费
        actual_value = shares * buy_price
        commission = actual_value * self.commission_rate

        total_cost = actual_value + commission

        if total_cost > self.cash:
            return

        # 更新资金
        self.cash -= total_cost

        # 记录持仓
        self.positions[stock_code] = PortfolioPosition(
            stock_code=stock_code,
            shares=shares,
            entry_price=buy_price,
            current_price=price,
            market_value=shares * price,
            weight=0,  # 稍后计算
            entry_date=date,
            days_held=0
        )

        # 记录交易
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'direction': 'buy',
            'price': buy_price,
            'shares': shares,
            'value': actual_value,
            'commission': commission,
            'score': score
        })

    def _sell_position(
        self,
        stock_code: str,
        price: float,
        date: datetime
    ):
        """卖出持仓"""
        if stock_code not in self.positions:
            return

        position = self.positions[stock_code]

        # 考虑滑点
        sell_price = price * (1 - self.slippage)

        # 计算卖出金额
        sell_value = position.shares * sell_price
        commission = sell_value * self.commission_rate

        actual_proceeds = sell_value - commission

        # 更新资金
        self.cash += actual_proceeds

        # 计算盈亏
        profit = sell_value - position.shares * position.entry_price

        # 记录交易
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'direction': 'sell',
            'price': sell_price,
            'shares': position.shares,
            'value': sell_value,
            'commission': commission,
            'profit': profit,
            'return_pct': profit / (position.shares * position.entry_price),
            'days_held': position.days_held
        })

        # 删除持仓
        del self.positions[stock_code]

    def _adjust_risk_parity_weights(
        self,
        date: datetime,
        stock_data_dict: Dict[str, pd.DataFrame]
    ):
        """调整风险平价权重"""
        if len(self.positions) <= 1:
            return

        # 计算各持仓的波动率
        volatilities = {}

        for stock_code in self.positions:
            df = stock_data_dict.get(stock_code)
            if df is not None and len(df) >= 20:
                returns = df['close'].pct_change().tail(20)
                vol = returns.std()
                volatilities[stock_code] = vol if vol > 0 else 0.02
            else:
                volatilities[stock_code] = 0.02

        # 计算风险平价权重
        inv_vol = {code: 1/vol for code, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        weights = {code: v/total_inv_vol for code, v in inv_vol.items()}

        # 更新持仓权重
        for stock_code in self.positions:
            self.positions[stock_code].weight = weights.get(stock_code, 0)

    def _update_positions(
        self,
        date: datetime,
        stock_data_dict: Dict[str, pd.DataFrame]
    ):
        """更新持仓市值"""
        total_value = self.cash

        for stock_code, position in self.positions.items():
            df = stock_data_dict.get(stock_code)
            if df is not None:
                if 'timestamp' in df.columns:
                    mask = pd.to_datetime(df['timestamp']).dt.date == date
                    day_df = df[mask]
                else:
                    day_df = df[df.index.date == date]

                if not day_df.empty:
                    position.current_price = day_df['close'].iloc[-1]
                    position.market_value = position.shares * position.current_price
                    position.days_held += 1

            total_value += position.market_value

        # 更新峰值
        if total_value > self.peak_value:
            self.peak_value = total_value

    def _record_snapshot(self, date: datetime):
        """记录快照"""
        total_value = self.cash + sum(p.market_value for p in self.positions.values())

        # 计算收益率
        if self.snapshots:
            prev_value = self.snapshots[-1].total_value
            daily_return = (total_value - prev_value) / prev_value
        else:
            daily_return = 0

        # 计算回撤
        drawdown = (self.peak_value - total_value) / self.peak_value

        snapshot = PortfolioSnapshot(
            date=date,
            total_value=total_value,
            cash=self.cash,
            positions=list(self.positions.values()),
            returns=daily_return,
            drawdown=drawdown
        )

        self.snapshots.append(snapshot)

    def _calculate_result(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> PortfolioBacktestResult:
        """计算回测结果"""
        if not self.snapshots:
            return PortfolioBacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                snapshots=[],
                trade_history=[]
            )

        # 计算指标
        final_capital = self.snapshots[-1].total_value
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # 年化收益
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 夏普比率
        returns = [s.returns for s in self.snapshots]
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0 else 0
        )

        # 最大回撤
        max_drawdown = max(s.drawdown for s in self.snapshots)

        # 胜率
        sell_trades = [t for t in self.trade_history if t['direction'] == 'sell']
        win_trades = [t for t in sell_trades if t.get('profit', 0) > 0]

        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0

        # 盈亏比
        total_profit = sum(t.get('profit', 0) for t in win_trades)
        total_loss = abs(sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0))

        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        return PortfolioBacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trade_history),
            snapshots=self.snapshots,
            trade_history=self.trade_history
        )


def run_portfolio_backtest(
    stock_data_dict: Dict[str, pd.DataFrame],
    score_calculator: Callable,
    **kwargs
) -> Dict:
    """
    便捷函数：运行组合回测

    Args:
        stock_data_dict: 股票数据字典
        score_calculator: 评分计算函数
        **kwargs: 其他参数

    Returns:
        Dict: 回测结果
    """
    engine = PortfolioBacktestEngine(**kwargs)
    result = engine.run_portfolio_backtest(stock_data_dict, score_calculator)

    return {
        'initial_capital': result.initial_capital,
        'final_capital': result.final_capital,
        'total_return': result.total_return,
        'annual_return': result.annual_return,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'total_trades': result.total_trades
    }