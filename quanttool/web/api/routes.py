"""API routes for QuantTool web application."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime
from ..schemas.experiment import ExperimentRunSchema
from ..application.backtest_service import BacktestService
from ..application.factor_service import FactorService
from ..application.data_service import DataService


router = APIRouter()


@router.get("/experiments")
async def list_experiments(
    run_type: str = None, status: str = None
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    runs = db.get_experiment_runs(run_type=run_type, status=status)

    return runs


@router.post("/backtest/run")
async def run_backtest(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a backtest experiment."""
    try:
        # Extract parameters from request
        strategy_name = request_data.get("strategy_name", "ma_cross")
        symbols = request_data.get("symbols", [])
        start_date_str = request_data.get("start_date", "2023-01-01")
        end_date_str = request_data.get("end_date", "2023-12-31")
        timeframe = request_data.get("timeframe", "10m")
        initial_cash = request_data.get("initial_cash", 100000.0)

        # Convert dates from strings
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # Initialize backtest service
        backtest_service = BacktestService()

        # Run backtest
        result = backtest_service.run_backtest(
            strategy_name=strategy_name,
            strategy_params=request_data.get("strategy_params", {}),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_cash=initial_cash,
            commission_rate=request_data.get("commission_rate", 0.0003),
            data_provider=request_data.get("data_provider", "tushare"),
        )

        # Convert result to dict (this is simplified)
        result_dict = {
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
        }

        # Save to metadata DB
        from ..infrastructure.stores.meta_db import MetaDB
        import uuid

        db = MetaDB()
        run_id = str(uuid.uuid4())
        db.save_experiment_run(
            {
                "id": run_id,
                "type": "backtest",
                "parameters": request_data,
                "git_commit": "unknown",  # In a real implementation, get from git
                "data_version": "v1.0",  # In a real implementation, track data versions
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "status": "completed",
                "results": result_dict,
                "artifacts": [],
            }
        )

        return {"run_id": run_id, "result": result_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")


@router.get("/backtest/runs/{run_id}")
async def get_backtest_result(run_id: str) -> Dict[str, Any]:
    """Get results for a specific backtest run."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    run = db.get_experiment_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")

    return run


@router.post("/factors/mine")
async def mine_factors(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mine factors across a universe of stocks."""
    try:
        # Extract parameters
        factor_name = request_data.get("factor_name", "momentum")
        symbols = request_data.get("symbols", [])
        start_date_str = request_data.get("start_date", "2023-01-01")
        end_date_str = request_data.get("end_date", "2023-12-31")

        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # Initialize factor service
        factor_service = FactorService()

        # Run factor mining
        results = factor_service.mine_factor(
            factor_name=factor_name,
            factor_params=request_data.get("factor_params", {}),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_provider=request_data.get("data_provider", "tushare"),
        )

        # Convert results to serializable format
        serialized_results = {}
        for symbol, result in results.items():
            serialized_results[symbol] = {
                "factor_name": result.factor_name,
                "ic": result.ic,
                "rank_ic": result.rank_ic,
                "win_rate": result.win_rate,
                "avg_return": result.avg_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
            }

        # Save to metadata DB
        from ..infrastructure.stores.meta_db import MetaDB
        import uuid

        db = MetaDB()
        run_id = str(uuid.uuid4())
        db.save_experiment_run(
            {
                "id": run_id,
                "type": "factor_mining",
                "parameters": request_data,
                "git_commit": "unknown",
                "data_version": "v1.0",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "status": "completed",
                "results": serialized_results,
                "artifacts": [],
            }
        )

        return {"run_id": run_id, "results": serialized_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mining factors: {str(e)}")


@router.get("/data/providers")
async def list_data_providers() -> List[str]:
    """List available data providers."""
    from ..core.registry import registry, ComponentType

    providers = registry.list_available(ComponentType.DATA_PROVIDER)
    return providers


@router.get("/strategies")
async def list_strategies() -> List[str]:
    """List available strategies."""
    from ..core.registry import registry, ComponentType

    strategies = registry.list_available(ComponentType.STRATEGY)
    return strategies


@router.get("/factors")
async def list_factors() -> List[str]:
    """List available factors."""
    from ..core.registry import registry, ComponentType

    factors = registry.list_available(ComponentType.FACTOR)
    return factors
