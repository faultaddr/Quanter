"""Backtest experiment lookup API routes."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException


router = APIRouter()

# ==================== 原有 API ====================

@router.get("/experiments")
async def list_experiments(
    run_type: str = None, status: str = None
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    from ....infrastructure.stores.meta_db_async import get_async_meta_db

    db = get_async_meta_db()
    runs = await db.get_experiment_runs(run_type=run_type, status=status)

    return runs


@router.get("/backtest/runs/{run_id}")
async def get_backtest_result(run_id: str) -> Dict[str, Any]:
    """Get results for a specific backtest run."""
    from ....infrastructure.stores.meta_db_async import get_async_meta_db

    db = get_async_meta_db()
    run = await db.get_experiment_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")

    return run
