from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_dict"):
        return result.as_dict()
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError(f"Unsupported result type: {type(result)}")


def _truncate_ids(ids: list[str], max_ids: int) -> str:
    if not ids:
        return ""
    if max_ids <= 0 or len(ids) <= max_ids:
        return ", ".join(ids)
    return ", ".join(ids[:max_ids]) + f", ... (+{len(ids) - max_ids})"


def solution_breakdown_tables(
    result: Any,
    *,
    max_agents: int = 30,
    max_task_ids: int = 16,
) -> dict[str, pd.DataFrame]:
    """
    Возвращает подробные таблицы решения:
    - summary: сводка по алгоритму
    - agents:  по агентам (выбранные задачи, тоннаж, км, часы)
    - tasks:   long-view задача -> агент
    """
    payload = _result_payload(result)
    algorithm = str(payload.get("algorithm", "unknown"))
    rows = payload.get("agent_solution_rows") or []
    if not isinstance(rows, list):
        rows = []

    if not rows:
        summary = pd.DataFrame(
            [
                {
                    "algorithm": algorithm,
                    "active_agents": 0,
                    "assigned_tasks": int(payload.get("assigned_routes", 0) or 0),
                    "total_mass_tons": 0.0,
                    "total_km": payload.get("total_km"),
                    "total_hours": payload.get("total_hours"),
                }
            ]
        )
        return {
            "summary": summary,
            "agents": pd.DataFrame(),
            "tasks": pd.DataFrame(),
        }

    agent_df = pd.DataFrame(rows).copy()
    if "tasks_count" not in agent_df.columns:
        agent_df["tasks_count"] = agent_df["task_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    agent_df = agent_df.sort_values(
        by=["tasks_count", "total_km", "agent_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "algorithm": algorithm,
                "active_agents": int(len(agent_df)),
                "assigned_tasks": int(agent_df["tasks_count"].sum()),
                "total_mass_tons": round(float(agent_df.get("total_mass_tons", pd.Series(dtype=float)).sum()), 3),
                "total_km": payload.get("total_km"),
                "total_hours": payload.get("total_hours"),
            }
        ]
    )

    task_rows: list[dict[str, Any]] = []
    for row in rows:
        aid = row.get("agent_id")
        task_ids = row.get("task_ids") or []
        if not isinstance(task_ids, list):
            continue
        for pos, tid in enumerate(task_ids, start=1):
            task_rows.append(
                {
                    "algorithm": algorithm,
                    "agent_id": aid,
                    "task_order": pos,
                    "task_id": tid,
                }
            )
    tasks_df = pd.DataFrame(task_rows)

    view = agent_df.copy()
    view["task_ids"] = view["task_ids"].apply(
        lambda ids: _truncate_ids(ids, max_task_ids) if isinstance(ids, list) else ""
    )
    view["route_ids"] = view["route_ids"].apply(
        lambda ids: _truncate_ids(ids, max_task_ids) if isinstance(ids, list) else ""
    )

    cols = [
        "agent_id",
        "vehicle_type",
        "is_compact",
        "tasks_count",
        "routes_count",
        "total_mass_tons",
        "total_km",
        "deadhead_km",
        "deadhead_share_pct",
        "total_hours",
        "task_ids",
    ]
    view = view[[c for c in cols if c in view.columns]]
    if max_agents > 0:
        view = view.head(max_agents)

    return {
        "summary": summary,
        "agents": view,
        "tasks": tasks_df,
    }
