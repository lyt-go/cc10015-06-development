"""预测模型 分析看板接口：汇总指标、参数运行与结果导出。

V006：训练批次与预测结果表落库后的查询、训练、续训、核对与全份导出接口。
"""
from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.schemas import EntryPayload
from app.services.forecast_model import RESULT_FIELDS, ForecastModelService
from app.store import store

router = APIRouter(prefix="/api/forecast_model", tags=["预测模型"])
service = ForecastModelService()
LIST_FIELDS = ["模型编号", "预测对象", "历史窗口", "预测周期", "模型类型", "参数配置", "训练批次", "评估误差"]
STATUSES = ["草稿", "训练中", "已就绪", "已废弃"]
BATCH_FIELDS = ["批次编号", "模型编号", "业务周期", "指标编号", "口径版本", "精度位数", "status",
                "冲突策略", "预测周期数", "已完成期次", "训练样本数", "评估误差", "拒绝点数", "来源版本"]
ARCHIVE_FIELDS = ["留档编号", "类型", "模型编号", "批次编号", "说明", "来源版本", "created_at"]


def csv_cell(value: Any) -> str:
    text = str(value) if value is not None else ""
    if text and text[0] in "=+-@":
        text = "'" + text
    return '"' + text.replace('"', '""') + '"'


def csv_response(rows: list[dict[str, Any]], headers: list[str], filename: str) -> StreamingResponse:
    buf = io.StringIO(newline="")
    buf.write("﻿")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", *headers])
    for row in rows:
        writer.writerow([csv_cell(row.get("id", "")), *[csv_cell(row.get(h, "")) for h in headers]])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    """汇总卡片读取落库数据，刷新不出现无来源数值。"""
    return {"metrics": [
        {"label": "训练批次", "value": len(store.rows("forecast_batch")), "trend": 0},
        {"label": "预测点总数", "value": len(store.rows("forecast_result")), "trend": 0},
        {"label": "留档记录", "value": len(store.rows("forecast_archive")), "trend": 0},
    ]}


@router.get("")
def list_models() -> dict[str, Any]:
    return {"items": service.list_models()}


@router.post("/run")
def run_analysis(payload: EntryPayload) -> dict[str, Any]:
    results = service.run_analysis(payload.values)
    return {"items": results}


@router.post("")
def create_entry(payload: EntryPayload) -> dict[str, Any]:
    """登记模型配置，缺必填字段时说明原因。"""
    entry, missing = service.create_entry(payload.values)
    if missing:
        return {"ok": False, "message": f"缺少必填字段：{'、'.join(missing)}"}
    return {"ok": True, "message": "模型配置已登记", "entry": entry}


@router.get("/results")
def list_forecast_results(
    模型编号: str | None = Query(default=None, description="按模型编号过滤"),
    批次编号: str | None = Query(default=None, description="按训练批次过滤"),
    消费场景: str | None = Query(default=None, description="默认最新批次；对比=展示并存批次"),
) -> dict[str, Any]:
    """预测结果表：每个预测点绑定训练批次与指标口径，默认唯一定位最新批次。"""
    items = service.list_results(模型编号=模型编号, 批次编号=批次编号, 消费场景=消费场景)
    return {"items": items, "total": len(items)}


@router.get("/results/export")
def export_forecast_results() -> StreamingResponse:
    """全份预测结果表：V006 完成后由接收方获取。"""
    return csv_response(service.export_rows(), RESULT_FIELDS + ["批次状态", "校验状态", "来源版本", "created_at"],
                        "forecast_result_v006.csv")


@router.get("/batches")
def list_batches(
    模型编号: str | None = Query(default=None),
    业务周期: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> dict[str, Any]:
    filters = {k: v for k, v in {"模型编号": 模型编号, "业务周期": 业务周期, "status": status}.items() if v}
    items = service.list_batches(filters)
    return {"items": items, "total": len(items)}


@router.get("/archives")
def list_archives() -> dict[str, Any]:
    """V006 留档：回滚清残留、批次归档、整批拒绝的核对记录。"""
    items = service.list_archives()
    return {"items": items, "total": len(items)}


@router.post("/consistency_check")
def consistency_check() -> dict[str, Any]:
    """一致性核对：发现残留或双批次事务异常时回滚清残留并按 V006 留档。"""
    return service.check_consistency()


@router.get("/export")
def export_entries() -> StreamingResponse:
    return csv_response(service.list_models(), LIST_FIELDS + ["status"], "forecast_model.csv")


@router.get("/{entry_id}")
def get_entry(entry_id: int) -> dict:
    """详情页：模型配置 + 训练批次 + 预测结果表，与工作台面板同源。"""
    detail = service.get_entry_detail(entry_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"模型配置 {entry_id} 不存在")
    return detail


@router.post("/{entry_id}/train")
def train(entry_id: int, payload: EntryPayload) -> dict[str, Any]:
    """训练并生成预测结果表：批次与结果同事务落库，越限点按阈值拒绝。"""
    return service.train(entry_id, payload.values)


@router.post("/{entry_id}/resume")
def resume(entry_id: int, payload: EntryPayload | None = None) -> dict[str, Any]:
    """续训：训练中断后只补写未完成周期，不覆盖上一完整预测批次。"""
    return service.resume(entry_id, payload.values if payload else None)
