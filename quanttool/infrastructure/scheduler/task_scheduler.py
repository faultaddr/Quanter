"""定时任务调度器 - 每日自动执行 scan、更新组合净值、生成报告."""

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from quanttool.core.logging import get_logger
from quanttool.infrastructure.stores.meta_db import MetaDB
from quanttool.application.portfolio_backtest_service import PortfolioBacktestService
from quanttool.infrastructure.data_providers.data_fetcher import EnhancedDataFetcher

logger = get_logger(__name__)


class DailyTaskScheduler:
    """每日定时任务调度器."""

    def __init__(
        self,
        db_path: str = "./quanttool.db",
        scan_hour: int = 15,
        scan_minute: int = 30,
        update_hour: int = 18,
        update_minute: int = 0,
        report_hour: int = 19,
        report_minute: int = 0,
    ):
        """
        初始化定时任务调度器.

        Args:
            db_path: 数据库路径
            scan_hour: 每日 scan 执行小时
            scan_minute: 每日 scan 执行分钟
            update_hour: 每日净值更新小时
            update_minute: 每日净值更新分钟
            report_hour: 每日报告生成小时
            report_minute: 每日报告生成分钟
        """
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "APScheduler is required. Install with: pip install apscheduler"
            )

        self.scheduler = AsyncIOScheduler()
        self.meta_db = MetaDB(db_path)
        self.backtest_service = PortfolioBacktestService(db_path)
        self.data_fetcher: Optional[EnhancedDataFetcher] = None

        self.scan_hour = scan_hour
        self.scan_minute = scan_minute
        self.update_hour = update_hour
        self.update_minute = update_minute
        self.report_hour = report_hour
        self.report_minute = report_minute

        self._scan_job_id = "daily_scan"
        self._update_job_id = "daily_update"
        self._report_job_id = "daily_report"
        self._running = False

    async def initialize(self):
        """初始化数据获取器."""
        self.data_fetcher = EnhancedDataFetcher()
        await self.data_fetcher.initialize()
        logger.info("Task scheduler initialized")

    def schedule_all_tasks(self):
        """调度所有定时任务."""
        # 每日 scan 任务 (15:30)
        self.scheduler.add_job(
            self.run_daily_scan,
            trigger=CronTrigger(hour=self.scan_hour, minute=self.scan_minute),
            id=self._scan_job_id,
            name="Daily Market Scan",
            replace_existing=True,
        )
        logger.info(f"Scheduled daily scan at {self.scan_hour:02d}:{self.scan_minute:02d}")

        # 每日净值更新任务 (18:00)
        self.scheduler.add_job(
            self.run_portfolio_update,
            trigger=CronTrigger(hour=self.update_hour, minute=self.update_minute),
            id=self._update_job_id,
            name="Daily Portfolio Update",
            replace_existing=True,
        )
        logger.info(f"Scheduled portfolio update at {self.update_hour:02d}:{self.update_minute:02d}")

        # 每日报告生成任务 (19:00)
        self.scheduler.add_job(
            self.run_report_generation,
            trigger=CronTrigger(hour=self.report_hour, minute=self.report_minute),
            id=self._report_job_id,
            name="Daily Report Generation",
            replace_existing=True,
        )
        logger.info(f"Scheduled report generation at {self.report_hour:02d}:{self.report_minute:02d}")

    def start(self):
        """启动调度器."""
        if not self._running:
            self.scheduler.start()
            self._running = True
            logger.info("Task scheduler started")

    def stop(self):
        """停止调度器."""
        if self._running:
            self.scheduler.shutdown()
            self._running = False
            logger.info("Task scheduler stopped")

    def is_running(self) -> bool:
        """检查调度器是否运行中."""
        return self._running

    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """获取已调度的任务列表."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            })
        return jobs

    async def run_daily_scan(self):
        """执行每日 scan 任务."""
        from ...factors.scoring_system import ScoringSystem
        from ...factors.stock_analyzer import StockAnalyzer

        execution_result = {"task": "daily_scan", "status": "started", "timestamp": datetime.now().isoformat()}

        try:
            today = date.today()

            # 检查是否是交易日
            if not await self._is_trading_day(today):
                logger.info(f"{today} is not a trading day, skipping scan")
                execution_result["status"] = "skipped"
                execution_result["reason"] = "not_trading_day"
                self.meta_db.record_task_execution({
                    "task_type": "daily_scan",
                    "status": "skipped",
                    "result": execution_result,
                })
                return

            logger.info(f"Starting daily scan for {today}")

            # 初始化组件
            analyzer = StockAnalyzer()
            scoring_system = ScoringSystem()

            # 获取沪深300成分股
            csi300_stocks = await self._get_csi300_constituents()
            logger.info(f"Scanning {len(csi300_stocks)} stocks from CSI300")

            # 分析每只股票
            results = []
            skipped = []

            for stock_info in csi300_stocks:
                try:
                    symbol = stock_info["symbol"]
                    name = stock_info["name"]

                    # 获取股票数据
                    df = analyzer.get_stock_data(symbol, days=360)
                    if df is None or len(df) < 60:
                        skipped.append({"symbol": symbol, "reason": "insufficient_data"})
                        continue

                    # 计算技术指标
                    df = analyzer.calculate_technical_indicators(df)

                    # 获取最新数据
                    latest = df.iloc[-1]
                    trade_date_T = latest.name.strftime("%Y-%m-%d")

                    # 计算评分
                    score_result = scoring_system.calculate_all_scores(
                        df, symbol, trade_date_T, trade_date_T, latest["close"]
                    )

                    if score_result:
                        results.append({
                            "symbol": symbol,
                            "name": name,
                            "close": latest["close"],
                            "daily_return": latest.get("daily_return", 0),
                            **score_result,
                        })

                except Exception as e:
                    logger.warning(f"Error analyzing {stock_info.get('symbol')}: {e}")
                    skipped.append({"symbol": stock_info.get('symbol'), "reason": str(e)})

            # 排序并取 Top 5
            results.sort(key=lambda x: x.get("total_score", 0), reverse=True)
            top_results = results[:5]

            # 保存 scan 记录
            scan_data = {
                "scan_date": today.isoformat(),
                "market": "csi300",
                "days_analyzed": 360,
                "total_stocks": len(csi300_stocks),
                "results": top_results,
            }
            scan_id = self.meta_db.save_scan_record(scan_data)

            logger.info(f"Scan completed. Top 5 stocks: {[r['symbol'] for r in top_results]}")

            # 自动创建投资组合回测
            if top_results:
                backtest_id = self.backtest_service.create_portfolio_from_scan(
                    scan_id=scan_id,
                    initial_capital=500000,
                    top_n=5,
                )
                logger.info(f"Created portfolio backtest: {backtest_id}")
                execution_result["backtest_id"] = backtest_id

            execution_result["status"] = "success"
            execution_result["scan_id"] = scan_id
            execution_result["top_stocks"] = [r['symbol'] for r in top_results]
            execution_result["scanned_count"] = len(csi300_stocks)
            execution_result["skipped_count"] = len(skipped)

            self.meta_db.record_task_execution({
                "task_type": "daily_scan",
                "status": "success",
                "result": execution_result,
            })

        except Exception as e:
            logger.error(f"Daily scan failed: {e}")
            execution_result["status"] = "failed"
            execution_result["error"] = str(e)
            self.meta_db.record_task_execution({
                "task_type": "daily_scan",
                "status": "failed",
                "error": str(e),
                "result": execution_result,
            })
            raise

    async def run_portfolio_update(self):
        """执行每日组合净值更新任务."""
        execution_result = {"task": "portfolio_update", "status": "started", "timestamp": datetime.now().isoformat()}

        try:
            today = date.today()

            # 获取所有活跃的组合
            active_portfolios = self.meta_db.get_active_portfolios()
            logger.info(f"Updating {len(active_portfolios)} active portfolios")

            updated_count = 0
            closed_count = 0

            for portfolio in active_portfolios:
                try:
                    backtest_id = portfolio["id"]
                    start_date = datetime.fromisoformat(portfolio["start_date"]).date()

                    # 计算持仓天数
                    holding_days = (today - start_date).days

                    # 检查是否需要平仓 (20个交易日)
                    if holding_days >= 20:
                        logger.info(f"Closing portfolio {backtest_id} after {holding_days} days")
                        self.backtest_service.close_portfolio(backtest_id, today)
                        closed_count += 1
                    else:
                        # 更新净值
                        self.backtest_service.update_portfolio_value(backtest_id, today)
                        updated_count += 1

                except Exception as e:
                    logger.warning(f"Error updating portfolio {portfolio.get('id')}: {e}")

            execution_result["status"] = "success"
            execution_result["updated_count"] = updated_count
            execution_result["closed_count"] = closed_count

            self.meta_db.record_task_execution({
                "task_type": "portfolio_update",
                "status": "success",
                "result": execution_result,
            })

            logger.info(f"Portfolio update completed: {updated_count} updated, {closed_count} closed")

        except Exception as e:
            logger.error(f"Portfolio update failed: {e}")
            execution_result["status"] = "failed"
            execution_result["error"] = str(e)
            self.meta_db.record_task_execution({
                "task_type": "portfolio_update",
                "status": "failed",
                "error": str(e),
                "result": execution_result,
            })
            raise

    async def run_report_generation(self):
        """执行每日报告生成任务."""
        from ...reports.daily_report_generator import DailyReportGenerator

        execution_result = {"task": "report_generation", "status": "started", "timestamp": datetime.now().isoformat()}

        try:
            today = date.today()
            logger.info(f"Generating daily report for {today}")

            # 生成报告
            report_gen = DailyReportGenerator(self.meta_db.db_path)
            report_content = report_gen.generate_daily_report(today)

            # 保存报告到文件
            report_dir = Path("./reports")
            report_dir.mkdir(exist_ok=True)
            report_file = report_dir / f"daily_report_{today.isoformat()}.md"
            report_file.write_text(report_content, encoding="utf-8")

            # 尝试发送邮件
            try:
                email_sent = report_gen.send_report_email(today, report_content)
                execution_result["email_sent"] = email_sent
            except Exception as e:
                logger.warning(f"Failed to send email report: {e}")
                execution_result["email_sent"] = False
                execution_result["email_error"] = str(e)

            execution_result["status"] = "success"
            execution_result["report_file"] = str(report_file)

            self.meta_db.record_task_execution({
                "task_type": "report_generation",
                "status": "success",
                "result": execution_result,
            })

            logger.info(f"Daily report generated: {report_file}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            execution_result["status"] = "failed"
            execution_result["error"] = str(e)
            self.meta_db.record_task_execution({
                "task_type": "report_generation",
                "status": "failed",
                "error": str(e),
                "result": execution_result,
            })
            raise

    async def _is_trading_day(self, check_date: date) -> bool:
        """检查是否是交易日."""
        # 周末不是交易日
        if check_date.weekday() >= 5:  # 5=周六, 6=周日
            return False

        # TODO: 添加节假日判断（可以通过 tushare 获取交易日历）
        return True

    async def _get_csi300_constituents(self) -> List[Dict[str, str]]:
        """获取沪深300成分股列表."""
        if self.data_fetcher:
            return await self.data_fetcher.get_csi300_constituents(include_names=True)

        # 降级方案：返回空列表，实际应该从数据库或文件加载
        logger.warning("Data fetcher not initialized, returning empty list")
        return []


class TaskSchedulerManager:
    """调度器管理器 - 用于启动和管理后台调度器."""

    _instance: Optional["TaskSchedulerManager"] = None
    _scheduler: Optional[DailyTaskScheduler] = None

    @classmethod
    def get_instance(cls) -> "TaskSchedulerManager":
        """获取单例实例."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_scheduler(self, db_path: str = "./quanttool.db") -> bool:
        """启动调度器."""
        try:
            if self._scheduler is None or not self._scheduler.is_running():
                self._scheduler = DailyTaskScheduler(db_path)
                asyncio.create_task(self._scheduler.initialize())
                self._scheduler.schedule_all_tasks()
                self._scheduler.start()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def stop_scheduler(self) -> bool:
        """停止调度器."""
        if self._scheduler and self._scheduler.is_running():
            self._scheduler.stop()
            self._scheduler = None
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态."""
        if self._scheduler and self._scheduler.is_running():
            return {
                "running": True,
                "jobs": self._scheduler.get_scheduled_jobs(),
            }
        return {"running": False, "jobs": []}
