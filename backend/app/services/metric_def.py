"""指标定义业务规则：状态流转、字段校验与筛选口径都收在这里。"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.store import store

MODULE = "metric_def"
LIST_FIELDS = ["指标编号", "指标名称", "计算公式", "统计粒度", "时间窗口", "基准取值", "精度位数", "口径版本"]
REQUIRED_FIELDS = ["指标编号", "指标名称", "计算公式"]
STATUS_ORDER = ["草稿", "待校验", "已生效", "已废弃"]
ACTION_RULES = {"提交校验": "待校验", "标记生效": "已生效", "标记废弃": "已废弃"}
NEGATIVE_ACTIONS = []


class MetricDefService:
    def list_entries(
        self,
        *,
        filters: dict[str, str] | None = None,
        keyword: str | None = None,
        status: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = store.rows(MODULE)
        for field, expected in (filters or {}).items():
            rows = [row for row in rows if expected.casefold() in str(row.get(field, "")).casefold()]
        if keyword:
            rows = [row for row in rows if keyword.casefold() in str(row.get("指标编号", "")).casefold()]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        total = len(rows)
        start = (page - 1) * size
        return rows[start:start + size], total

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        return store.find(MODULE, entry_id)

    def create_entry(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        missing = [field for field in REQUIRED_FIELDS if not str(values.get(field) or "").strip()]
        if missing:
            return None, missing
        rows = store.rows(MODULE)
        entry = {"id": max((int(row.get("id", 0)) for row in rows), default=0) + 1}
        entry.update({field: values.get(field) for field in LIST_FIELDS})
        entry["status"] = STATUS_ORDER[0]
        entry["pending"] = True
        entry["abnormal"] = False
        rows.append(entry)
        return entry, []

    def run_action(self, entry_id: int, action: str) -> tuple[dict[str, Any] | None, str]:
        entry = store.find(MODULE, entry_id)
        if entry is None:
            return None, f"指标定义 {entry_id} 不存在或已归档"
        if action not in ACTION_RULES:
            return None, f"动作「{action}」不属于指标定义可执行范围"
        target = ACTION_RULES[action]
        if target not in STATUS_ORDER:
            return None, f"目标状态「{target}」不在允许的状态序列里"
        entry["status"] = target
        entry["pending"] = target != STATUS_ORDER[-1]
        entry["abnormal"] = action in NEGATIVE_ACTIONS
        return entry, f"指标定义已{action}"

    def validate_forecast_precision(self) -> dict[str, Any]:
        """V006 跨模块联动：读取预测结果表的指标标识和精度位数，校验预测值展示精度。"""
        rows = store.rows("forecast_result")
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get("指标编号") or "未绑定指标"), []).append(row)
        items: list[dict[str, Any]] = []
        for code, group in sorted(groups.items()):
            metric = store.find_by_field(MODULE, "指标编号", code)
            defined_precision: int | None = None
            if metric is not None:
                try:
                    defined_precision = int(str(metric.get("精度位数")).strip())
                except (TypeError, ValueError):
                    defined_precision = None
            table_precisions = sorted({int(r.get("精度位数", -1)) for r in group})
            over_limit = 0
            for r in group:
                precision = int(r.get("精度位数", 4))
                exponent = Decimal(str(r.get("预测值", "0"))).as_tuple().exponent
                decimals = max(0, -exponent) if isinstance(exponent, int) else 0
                if decimals > precision:
                    over_limit += 1
            issues: list[str] = []
            if metric is None:
                issues.append("指标定义不存在")
            if defined_precision is not None and any(p != defined_precision for p in table_precisions):
                issues.append(f"结果表精度{table_precisions}与指标定义精度{defined_precision}不一致")
            if over_limit:
                issues.append(f"{over_limit}个预测值展示精度超过精度位数")
            items.append({
                "指标编号": code,
                "指标名称": (metric or {}).get("指标名称", "—"),
                "口径版本": group[-1].get("口径版本", ""),
                "结果表精度": table_precisions,
                "定义精度": defined_precision if defined_precision is not None else "—",
                "预测点数": len(group),
                "精度超限点": over_limit,
                "校验结论": "通过" if not issues else "；".join(issues),
            })
        return {"items": items, "total": len(items)}
