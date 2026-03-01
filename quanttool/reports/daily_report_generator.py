"""每日投资报告生成器."""

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from quanttool.infrastructure.stores.meta_db import MetaDB
from quanttool.application.portfolio_backtest_service import PortfolioBacktestService
from quanttool.infrastructure.notification.email_service import EmailService


class DailyReportGenerator:
    """每日投资报告生成器 - 汇总 scan 结果和投资组合表现."""

    def __init__(self, db_path: str = "./quanttool.db"):
        self.db = MetaDB(db_path)
        self.backtest_service = PortfolioBacktestService(db_path)

    def generate_daily_report(self, report_date: Optional[date] = None) -> str:
        """生成完整每日报告.

        Args:
            report_date: 报告日期，默认为今天

        Returns:
            Markdown 格式的报告
        """
        if report_date is None:
            report_date = date.today()

        date_str = report_date.strftime("%Y-%m-%d")

        # 获取当日 scan 记录
        scan_history = self.db.get_scan_history(start_date=date_str, end_date=date_str, limit=1)

        report_lines = [
            f"# 每日投资报告 - {date_str}",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        # 1. 当日 Scan 结果
        if scan_history:
            scan_section = self._generate_scan_section(scan_history[0])
            report_lines.extend(scan_section)

        # 2. 活跃组合表现
        report_lines.extend(["## 活跃投资组合", ""])
        active_section = self._generate_active_portfolios_section()
        report_lines.extend(active_section)

        # 3. 历史组合回顾（最近平仓的）
        report_lines.extend(["## 近期平仓组合回顾", ""])
        closed_section = self._generate_closed_portfolios_section()
        report_lines.extend(closed_section)

        # 4. 策略有效性分析
        report_lines.extend(["## 策略有效性分析", ""])
        analysis_section = self._generate_strategy_analysis()
        report_lines.extend(analysis_section)

        return "\n".join(report_lines)

    def _generate_scan_section(self, scan_record: Dict) -> List[str]:
        """生成 scan 结果部分."""
        lines = [
            "## 当日 Scan 结果",
            "",
            f"**扫描日期**: {scan_record.get('scan_date')}",
            f"**市场**: {scan_record.get('market', 'CSI300')}",
            f"**扫描股票数**: {scan_record.get('total_stocks', 0)}",
            "",
        ]

        # 获取完整 scan 结果（包含个股详情）
        full_record = self.db.get_scan_record(scan_record.get('id'))
        if full_record and full_record.get('results'):
            lines.extend(["### Top 5 推荐股票", ""])
            lines.append("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 评级 | 操作建议 | 风险等级 |")
            lines.append("|------|------|------|--------|------|------|----------|----------|")

            for stock in full_record['results'][:5]:
                lines.append(
                    f"| {stock.get('rank', '-')} | "
                    f"{stock.get('symbol', '-')} | "
                    f"{stock.get('name', '-')} | "
                    f"{stock.get('close', 0):.2f} | "
                    f"{stock.get('total_score', 0)} | "
                    f"{stock.get('rating', '-')} | "
                    f"{stock.get('action', '-')} | "
                    f"{stock.get('risk_level', '-')} |"
                )

            lines.append("")

            # 评分分布
            lines.extend(["### 评分分布", ""])
            scores = [s.get('total_score', 0) for s in full_record['results']]
            if scores:
                lines.append(f"- 最高评分: {max(scores)}")
                lines.append(f"- 最低评分: {min(scores)}")
                lines.append(f"- 平均评分: {sum(scores)/len(scores):.1f}")
            lines.append("")

        return lines

    def _generate_active_portfolios_section(self) -> List[str]:
        """生成活跃组合表现部分."""
        lines = []
        active_portfolios = self.db.get_active_portfolios()

        if not active_portfolios:
            lines.append("*当前没有活跃的投资组合*")
            lines.append("")
            return lines

        for portfolio in active_portfolios:
            backtest_id = portfolio.get('id')
            full_data = self.db.get_portfolio_backtest(backtest_id)

            if not full_data:
                continue

            # 计算当前表现
            initial_capital = full_data.get('initial_capital', 500000)
            daily_values = full_data.get('daily_values', [])

            if daily_values:
                current_value = daily_values[-1].get('total_value', initial_capital)
                total_return = (current_value - initial_capital) / initial_capital * 100
                days_held = len(daily_values)
            else:
                current_value = initial_capital
                total_return = 0
                days_held = 0

            lines.append(f"### {portfolio.get('portfolio_name', backtest_id[:8])}")
            lines.append("")
            lines.append(f"- **初始资金**: {initial_capital:,.0f} 元")
            lines.append(f"- **当前市值**: {current_value:,.2f} 元")
            lines.append(f"- **累计收益**: {total_return:+.2f}%")
            lines.append(f"- **持仓天数**: {days_held} 天")
            lines.append("")

            # 持仓明细
            holdings = full_data.get('holdings', [])
            if holdings:
                lines.append("**持仓明细**:")
                lines.append("")
                lines.append("| 代码 | 名称 | 买入价 | 数量 | 状态 |")
                lines.append("|------|------|--------|------|------|")
                for h in holdings:
                    lines.append(
                        f"| {h.get('symbol', '-')} | "
                        f"{h.get('name', '-')} | "
                        f"{h.get('entry_price', 0):.2f} | "
                        f"{h.get('shares', 0)} | "
                        f"{h.get('status', '-')} |"
                    )
                lines.append("")

        return lines

    def _generate_closed_portfolios_section(self) -> List[str]:
        """生成已平仓组合回顾部分."""
        lines = []

        # 获取最近平仓的组合（从数据库查询）
        conn = self.db._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, portfolio_name, initial_capital, total_return,
                   sharpe_ratio, max_drawdown, start_date, end_date
            FROM portfolio_backtests
            WHERE status = 'closed'
            ORDER BY end_date DESC
            LIMIT 5
            """
        )

        closed_portfolios = cursor.fetchall()
        conn.close()

        if not closed_portfolios:
            lines.append("*暂无已平仓的组合*")
            lines.append("")
            return lines

        lines.append("| 组合名称 | 初始资金 | 总收益 | 夏普比率 | 最大回撤 | 持仓周期 |")
        lines.append("|----------|----------|--------|----------|----------|----------|")

        for row in closed_portfolios:
            portfolio_name = row[1] or row[0][:8]
            initial = row[2]
            total_return = row[3] or 0
            sharpe = row[4] or 0
            max_dd = row[5] or 0
            start = row[6] or '-'
            end = row[7] or '-'

            lines.append(
                f"| {portfolio_name} | "
                f"{initial:,.0f} | "
                f"{total_return:+.2f}% | "
                f"{sharpe:.2f} | "
                f"{max_dd:.2f}% | "
                f"{start} ~ {end} |"
            )

        lines.append("")
        return lines

    def _generate_strategy_analysis(self) -> List[str]:
        """生成策略有效性分析."""
        lines = []

        # 获取所有已平仓组合的统计数据
        conn = self.db._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*), AVG(total_return),
                   SUM(CASE WHEN total_return > 0 THEN 1 ELSE 0 END),
                   AVG(sharpe_ratio), AVG(max_drawdown)
            FROM portfolio_backtests
            WHERE status = 'closed' AND total_return IS NOT NULL
            """
        )

        result = cursor.fetchone()
        conn.close()

        if result and result[0] > 0:
            total_count = result[0]
            avg_return = result[1] or 0
            win_count = result[2] or 0
            avg_sharpe = result[3] or 0
            avg_drawdown = result[4] or 0
            win_rate = win_count / total_count * 100 if total_count > 0 else 0

            lines.append(f"### 历史回测统计 (样本数: {total_count})")
            lines.append("")
            lines.append(f"- **平均收益率**: {avg_return:+.2f}%")
            lines.append(f"- **胜率**: {win_rate:.1f}% ({win_count}/{total_count})")
            lines.append(f"- **平均夏普比率**: {avg_sharpe:.2f}")
            lines.append(f"- **平均最大回撤**: {avg_drawdown:.2f}%")
            lines.append("")

            # 简单评价
            if win_rate > 60 and avg_return > 0:
                lines.append("✅ **策略评价**: 表现良好，值得继续关注")
            elif win_rate > 50 and avg_return > 0:
                lines.append("⚠️ **策略评价**: 表现尚可，需持续观察")
            else:
                lines.append("❌ **策略评价**: 表现欠佳，建议优化策略")
            lines.append("")
        else:
            lines.append("*暂无足够的历史数据进行策略分析*")
            lines.append("")

        return lines

    def generate_and_save_report(self, report_date: Optional[date] = None, output_dir: str = "./reports") -> str:
        """生成并保存报告到文件.

        Returns:
            保存的文件路径
        """
        import os

        report = self.generate_daily_report(report_date)

        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        date_str = (report_date or date.today()).strftime("%Y%m%d")
        filename = f"daily_report_{date_str}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        return filepath

    async def generate_and_send_report(
        self,
        report_date: Optional[date] = None,
        email_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """生成报告并发送邮件.

        Args:
            report_date: 报告日期
            email_config: 邮件配置，如果为 None 则从数据库获取

        Returns:
            是否发送成功
        """
        # 生成报告
        report = self.generate_daily_report(report_date)

        # 获取邮件配置
        if email_config is None:
            email_config = self.db.get_email_config()

        if not email_config:
            print("未配置邮件发送")
            return False

        # 创建邮件服务并发送
        email_service = EmailService(
            smtp_host=email_config.get('smtp_host', 'smtp.gmail.com'),
            smtp_port=email_config.get('smtp_port', 587),
            username=email_config.get('username', ''),
            password=email_config.get('password', ''),
        )

        report_date_str = (report_date or date.today()).strftime("%Y-%m-%d")

        try:
            await email_service.send_daily_report(
                report_date=report_date or date.today(),
                report_content=report,
                recipients=email_config.get('to_addrs', []),
            )
            return True
        except Exception as e:
            print(f"发送邮件失败: {e}")
            return False
