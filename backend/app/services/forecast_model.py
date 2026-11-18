"""预测模型 分析看板服务：汇总指标、运行分析并生成模拟结果。

V006 预测批次与预测结果表：
- 训练批次（forecast_batch）与预测结果表（forecast_result）同事务落库，刷新可回溯唯一来源；
- 每个预测点绑定训练批次与指标口径（指标编号、口径版本、精度位数）；
- 预测值须满足口径规则与阈值条件，超出范围或越过上限时按阈值拒绝，整批被拒则回滚并留档；
- 同一模型同一业务周期重复训练产生版本冲突：默认最新批次替换并归档旧批次，
  可选双批并存、按消费场景读取；
- 训练中断后只能补写未完成周期，不能覆盖上一完整预测批次；
- 一致性核对发现残留或双批次事务异常时回滚清残留，并按 V006 留档（forecast_archive）核对。
"""
from __future__ import annotations

import copy
import json
import random
import time
from typing import Any

from app.store import store

MODULE = "forecast_model"
BATCH_TABLE = "forecast_batch"
RESULT_TABLE = "forecast_result"
ARCHIVE_TABLE = "forecast_archive"
SPEC = "V006"
REQUIRED_FIELDS = ["模型编号", "预测对象", "历史窗口"]
STATUS_ORDER = ["草稿", "训练中", "已就绪", "已废弃"]
ACTION_RULES = {"开始训练": "训练中", "确认就绪": "已就绪", "废弃模型": "已废弃"}
NEGATIVE_ACTIONS = []
BATCH_ACTIVE_STATUSES = ("训练中", "已完成", "已中断")
CONFLICT_POLICIES = ("replace", "coexist")
RESULT_FIELDS = ["批次编号", "模型编号", "指标编号", "口径版本", "精度位数", "业务周期", "预测期次", "预测值"]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_history(raw: Any) -> list[float]:
    """历史观测数据既接受数组，也接受逗号分隔的字符串。"""
    if raw is None:
        return []
    parts = raw if isinstance(raw, (list, tuple)) else str(raw).replace("，", ",").split(",")
    values: list[float] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values


def _validate_point(value: float, lower: float, upper: float | None) -> str | None:
    """口径规则与阈值条件：返回 None 表示通过，否则给出拒绝原因。"""
    if value != value or value in (float("inf"), float("-inf")):
        return "预测值非有限数，违反口径规则"
    if value < lower:
        return f"低于阈值下限{lower}"
    if upper is not None and value > upper:
        return f"越过阈值上限{upper}"
    return None


class ForecastModelService:
    # ------------------------------------------------------------------
    # 模型配置（既有能力）
    # ------------------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        """模型清单直接读落库数据，刷新不叠加无来源数值。"""
        return [dict(row) for row in store.rows(MODULE)]

    def run_analysis(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        count = int(params.get("样本范围", params.get("并行数量", "6")))
        results: list[dict] = []
        for i in range(count):
            results.append({
                "id": i + 1,
                "分组": f"分组{(i%3)+1}",
                "指标值": round(random.uniform(0.62, 1.02), 3),
                "对比基准": round(random.uniform(0.58, 0.95), 3),
                "差异": round(random.uniform(-0.2, 0.3), 3),
            })
        return results

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        return store.find(MODULE, entry_id)

    def get_entry_detail(self, entry_id: int) -> dict[str, Any] | None:
        """详情页数据：模型配置 + 训练批次 + 预测结果表，与工作台面板同源。"""
        entry = self.get_entry(entry_id)
        if entry is None:
            return None
        model_code = str(entry.get("模型编号", ""))
        detail = dict(entry)
        detail["batches"] = [dict(b) for b in store.rows(BATCH_TABLE) if b.get("模型编号") == model_code]
        detail["results"] = self.list_results(模型编号=model_code)
        return detail

    def create_entry(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        missing = [f for f in REQUIRED_FIELDS if not str(values.get(f) or "").strip()]
        if missing:
            return None, missing
        rows = store.rows(MODULE)
        entry = {"id": max((int(r.get("id", 0)) for r in rows), default=0) + 1}
        for f in REQUIRED_FIELDS + ["预测周期", "模型类型", "参数配置", "训练批次", "评估误差"]:
            entry[f] = values.get(f, "")
        entry["status"] = STATUS_ORDER[0]
        entry["pending"] = True
        entry["abnormal"] = False
        rows.append(entry)
        return entry, []

    # ------------------------------------------------------------------
    # 预测值计算（既有瞬时能力保留，供兼容与内部复用）
    # ------------------------------------------------------------------
    def _predict_values(self, entry: dict[str, Any], values: list[float]) -> list[float]:
        """按模型配置计算多周期预测值（未舍入的原始值）。"""
        model_type = str(entry.get("模型类型", "moving_average"))
        try:
            predict_periods = int(entry.get("预测周期", "3"))
        except Exception:
            predict_periods = 3
        predict_periods = max(1, min(predict_periods, 30))
        predictions: list[float] = []
        if "moving" in model_type.lower():
            try:
                window = int(str(entry.get("参数配置", "window=3")).split("window=")[-1].split(",")[0])
            except Exception:
                window = 3
            window = max(1, window)
            series = list(values)
            for _ in range(predict_periods):
                pred = sum(series[-window:]) / window
                predictions.append(pred)
                series = series + [pred]
        elif "linear" in model_type.lower() or "regression" in model_type.lower():
            n = len(values)
            x_mean = (n - 1) / 2
            y_mean = sum(values) / n
            num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denom = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / denom if denom != 0 else 0
            intercept = y_mean - slope * x_mean
            for i in range(predict_periods):
                predictions.append(slope * (n + i) + intercept)
        else:
            mean = sum(values) / len(values)
            predictions = [mean for _ in range(predict_periods)]
        return predictions

    def train_and_predict(self, entry_id: int, values: list[float]) -> dict[str, Any]:
        """对历史数据执行简单移动平均或线性回归预测（瞬时数组，不落库）。"""
        entry = self.get_entry(entry_id)
        if not entry:
            return {"ok": False, "message": "模型不存在"}
        if len(values) < 2:
            return {"ok": False, "message": "历史数据不足"}
        predictions = [round(p, 4) for p in self._predict_values(entry, list(values))]
        return {"ok": True, "predictions": predictions, "method": entry.get("模型类型", "moving_average"),
                "message": f"预测完成，{len(predictions)}个周期"}

    # ------------------------------------------------------------------
    # V006：口径解析、阈值解析、事务与留档
    # ------------------------------------------------------------------
    def _resolve_caliber(self, entry: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """指标口径：优先显式指标编号，其次按预测对象匹配指标定义。"""
        metric_code = str(params.get("指标编号") or "").strip()
        target = str(entry.get("预测对象") or "").strip()
        matched = store.find_by_field("metric_def", "指标编号", metric_code) if metric_code else None
        if matched is None and target:
            for row in store.rows("metric_def"):
                if target in (str(row.get("指标编号", "")), str(row.get("指标名称", ""))):
                    matched = row
                    break
        precision = 4
        caliber_version = "v1"
        code = metric_code or target or "未绑定指标"
        if matched is not None:
            code = str(matched.get("指标编号") or code)
            caliber_version = str(matched.get("口径版本") or "v1")
            try:
                precision = int(str(matched.get("精度位数")).strip())
            except Exception:
                precision = 4
        return {"指标编号": code, "口径版本": caliber_version, "精度位数": max(0, min(precision, 10))}

    def _resolve_thresholds(self, entry: dict[str, Any], params: dict[str, Any]) -> tuple[float, float | None]:
        """阈值条件：请求参数优先，其次模型参数配置里的 lower=/upper=，默认下限 0。"""
        cfg = str(entry.get("参数配置", "") or "")

        def from_cfg(key: str) -> str | None:
            for part in cfg.split(","):
                part = part.strip()
                if part.startswith(f"{key}="):
                    return part.split("=", 1)[1]
            return None

        lower_raw = params.get("阈值下限", from_cfg("lower"))
        upper_raw = params.get("阈值上限", from_cfg("upper"))
        lower = 0.0
        if lower_raw not in (None, ""):
            try:
                lower = float(lower_raw)
            except (TypeError, ValueError):
                lower = 0.0
        upper: float | None = None
        if upper_raw not in (None, ""):
            try:
                upper = float(upper_raw)
            except (TypeError, ValueError):
                upper = None
        return lower, upper

    def _snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {t: copy.deepcopy(store.rows(t)) for t in (BATCH_TABLE, RESULT_TABLE)}

    def _restore(self, snap: dict[str, list[dict[str, Any]]]) -> None:
        for table, rows in snap.items():
            store.rows(table)[:] = rows

    def _archive(self, kind: str, model_code: str, batch_code: str, note: str,
                 snapshot_rows: Any) -> dict[str, Any]:
        """V006 留档：回滚清残留、批次归档、整批拒绝都写入 forecast_archive 供核对。"""
        rows = store.rows(ARCHIVE_TABLE)
        record = {
            "id": store.next_id(ARCHIVE_TABLE),
            "留档编号": f"AR-{store.next_id(ARCHIVE_TABLE):04d}",
            "类型": kind,
            "模型编号": model_code,
            "批次编号": batch_code,
            "说明": note,
            "快照": json.dumps(snapshot_rows, ensure_ascii=False, default=str),
            "来源版本": SPEC,
            "created_at": _now(),
        }
        rows.append(record)
        return record

    def _archive_batch(self, batch: dict[str, Any], reason: str) -> None:
        """旧批次归档：结果行移出预测结果表，批次状态置为已归档并留档。"""
        code = str(batch.get("批次编号", ""))
        removed = [r for r in store.rows(RESULT_TABLE) if r.get("批次编号") == code]
        store.rows(RESULT_TABLE)[:] = [r for r in store.rows(RESULT_TABLE) if r.get("批次编号") != code]
        batch["status"] = "已归档"
        batch["finished_at"] = _now()
        self._archive("批次归档", str(batch.get("模型编号", "")), code, reason,
                      {"batch": dict(batch), "results": removed})

    def _in_sample_mae(self, entry: dict[str, Any], values: list[float]) -> Any:
        """评估误差：历史样本内一步回测的平均绝对误差。"""
        try:
            model_type = str(entry.get("模型类型", "moving_average")).lower()
            if "moving" in model_type:
                try:
                    window = int(str(entry.get("参数配置", "window=3")).split("window=")[-1].split(",")[0])
                except Exception:
                    window = 3
                window = max(1, window)
                errs = [abs(values[i] - sum(values[i - window:i]) / window)
                        for i in range(window, len(values))]
            elif "linear" in model_type or "regression" in model_type:
                n = len(values)
                x_mean = (n - 1) / 2
                y_mean = sum(values) / n
                num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
                denom = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / denom if denom != 0 else 0
                intercept = y_mean - slope * x_mean
                errs = [abs(values[i] - (slope * i + intercept)) for i in range(n)]
            else:
                mean = sum(values) / len(values)
                errs = [abs(v - mean) for v in values]
            return round(sum(errs) / len(errs), 4) if errs else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # V006：训练落库、续训、查询、核对
    # ------------------------------------------------------------------
    def train(self, entry_id: int, params: dict[str, Any]) -> dict[str, Any]:
        """读取模型配置与历史观测数据，计算多周期预测值并生成预测结果表。

        训练批次与预测结果同事务落库；越限预测点按阈值拒绝；同业务周期重复训练
        按冲突策略处理（replace 替换并归档旧批次 / coexist 双批并存）。
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return {"ok": False, "message": f"模型配置 {entry_id} 不存在"}
        history = _parse_history(params.get("历史数据"))
        if len(history) < 2:
            return {"ok": False, "message": "历史数据不足，至少需要2个观测点"}
        model_code = str(entry.get("模型编号") or f"MODEL-{entry_id}")
        period = str(params.get("业务周期") or _now()[:7]).strip()
        policy = str(params.get("冲突策略") or "replace").strip()
        if policy not in CONFLICT_POLICIES:
            return {"ok": False, "message": f"冲突策略「{policy}」不支持，可选：replace、coexist"}
        incomplete = [b for b in store.rows(BATCH_TABLE)
                      if b.get("模型编号") == model_code and b.get("业务周期") == period
                      and b.get("status") in ("训练中", "已中断")]
        if incomplete:
            return {"ok": False,
                    "message": f"批次 {incomplete[-1]['批次编号']} 尚未完成，训练中断后只能续训补写未完成周期",
                    "batch": dict(incomplete[-1])}
        snap = self._snapshot()
        archived: list[str] = []
        try:
            completes = [b for b in store.rows(BATCH_TABLE)
                         if b.get("模型编号") == model_code and b.get("业务周期") == period
                         and b.get("status") == "已完成"]
            if completes and policy == "replace":
                for old in completes:
                    self._archive_batch(old, "同业务周期重复训练产生版本冲突，按V006以最新训练批次替换并归档")
                    archived.append(str(old.get("批次编号", "")))
            caliber = self._resolve_caliber(entry, params)
            lower, upper = self._resolve_thresholds(entry, params)
            raw = self._predict_values(entry, list(history))
            periods = len(raw)
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for idx, value in enumerate(raw, start=1):
                reason = _validate_point(value, lower, upper)
                if reason:
                    rejected.append({"预测期次": idx, "预测值": round(value, 6), "拒绝原因": reason})
                else:
                    accepted.append({"预测期次": idx, "预测值": round(value, caliber["精度位数"])})
            interrupt_raw = str(params.get("模拟中断期次") or "").strip()
            interrupt_after = int(interrupt_raw) if interrupt_raw else None
            interrupted = interrupt_after is not None and interrupt_after < len(accepted)
            if interrupt_after is not None:
                accepted = accepted[:max(0, interrupt_after)]
            if not accepted:
                self._restore(snap)
                self._archive("整批拒绝", model_code, "",
                              f"业务周期{period}全部{periods}个预测点越限被阈值拒绝，事务回滚未落库",
                              {"rejected": rejected})
                return {"ok": False,
                        "message": f"全部{periods}个预测点超出范围或越过上限，按阈值拒绝并已回滚",
                        "rejected": rejected}
            batch = {
                "id": store.next_id(BATCH_TABLE),
                "批次编号": f"FB-{store.next_id(BATCH_TABLE):04d}",
                "模型编号": model_code,
                "model_id": entry_id,
                "业务周期": period,
                "指标编号": caliber["指标编号"],
                "口径版本": caliber["口径版本"],
                "精度位数": caliber["精度位数"],
                "status": "已中断" if interrupted else "已完成",
                "冲突策略": policy,
                "预测周期数": periods,
                "已完成期次": len(accepted),
                "训练样本数": len(history),
                "评估误差": self._in_sample_mae(entry, history),
                "阈值下限": lower,
                "阈值上限": upper if upper is not None else "",
                "拒绝点数": len(rejected),
                "拒绝明细": rejected,
                "历史数据": list(history),
                "来源版本": SPEC,
                "created_at": _now(),
                "finished_at": "" if interrupted else _now(),
            }
            store.rows(BATCH_TABLE).append(batch)
            for item in accepted:
                store.rows(RESULT_TABLE).append({
                    "id": store.next_id(RESULT_TABLE),
                    "批次编号": batch["批次编号"],
                    "模型编号": model_code,
                    "指标编号": caliber["指标编号"],
                    "口径版本": caliber["口径版本"],
                    "精度位数": caliber["精度位数"],
                    "业务周期": period,
                    "预测期次": item["预测期次"],
                    "预测值": item["预测值"],
                    "校验状态": "通过",
                    "来源版本": SPEC,
                    "created_at": _now(),
                })
            entry["训练批次"] = batch["批次编号"]
            entry["评估误差"] = batch["评估误差"]
            entry["status"] = "训练中" if interrupted else "已就绪"
            entry["pending"] = interrupted
        except Exception as exc:  # 事务异常：回滚并留档
            self._restore(snap)
            self._archive("回滚清残留", model_code, "", f"训练事务异常已回滚：{exc}", {})
            return {"ok": False, "message": f"训练事务异常已回滚：{exc}"}
        check = self.check_consistency()
        if interrupted:
            message = f"训练中断，批次{batch['批次编号']}已完成{len(accepted)}/{periods}期，续训可补写未完成周期"
        else:
            message = f"训练完成，批次{batch['批次编号']}写入{len(accepted)}个预测点"
            if rejected:
                message += f"，{len(rejected)}个越限点按阈值拒绝"
            if archived:
                message += f"，旧批次{'、'.join(archived)}已归档"
        return {"ok": True, "message": message, "batch": dict(batch),
                "accepted": accepted, "rejected": rejected, "archived": archived, "check": check}

    def resume(self, entry_id: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """续训：只补写中断批次的未完成周期，不覆盖上一完整预测批次。"""
        entry = self.get_entry(entry_id)
        if not entry:
            return {"ok": False, "message": f"模型配置 {entry_id} 不存在"}
        model_code = str(entry.get("模型编号") or f"MODEL-{entry_id}")
        period = str((params or {}).get("业务周期") or "").strip()
        candidates = [b for b in store.rows(BATCH_TABLE)
                      if b.get("模型编号") == model_code and b.get("status") in ("训练中", "已中断")
                      and (not period or b.get("业务周期") == period)]
        if not candidates:
            return {"ok": False, "message": "没有待补写的中断批次"}
        batch = candidates[-1]
        snap = self._snapshot()
        try:
            history = [float(x) for x in batch.get("历史数据", [])]
            raw = self._predict_values(entry, list(history))
            done = int(batch.get("已完成期次", 0))
            lower = float(batch.get("阈值下限", 0.0) or 0.0)
            upper_raw = batch.get("阈值上限", "")
            upper = float(upper_raw) if upper_raw not in (None, "") else None
            precision = int(batch.get("精度位数", 4))
            rejected = list(batch.get("拒绝明细", []))
            appended: list[dict[str, Any]] = []
            for offset, value in enumerate(raw[done:], start=done + 1):
                reason = _validate_point(value, lower, upper)
                if reason:
                    rejected.append({"预测期次": offset, "预测值": round(value, 6), "拒绝原因": reason})
                    continue
                row = {
                    "id": store.next_id(RESULT_TABLE),
                    "批次编号": batch["批次编号"],
                    "模型编号": model_code,
                    "指标编号": batch.get("指标编号", ""),
                    "口径版本": batch.get("口径版本", ""),
                    "精度位数": precision,
                    "业务周期": batch.get("业务周期", ""),
                    "预测期次": offset,
                    "预测值": round(value, precision),
                    "校验状态": "通过",
                    "来源版本": SPEC,
                    "created_at": _now(),
                }
                store.rows(RESULT_TABLE).append(row)
                appended.append(dict(row))
            batch["已完成期次"] = int(batch.get("预测周期数", len(raw)))
            batch["拒绝明细"] = rejected
            batch["拒绝点数"] = len(rejected)
            batch["status"] = "已完成"
            batch["finished_at"] = _now()
            if entry.get("status") == "训练中":
                entry["status"] = "已就绪"
                entry["pending"] = False
        except Exception as exc:
            self._restore(snap)
            self._archive("回滚清残留", model_code, str(batch.get("批次编号", "")),
                          f"续训事务异常已回滚：{exc}", {})
            return {"ok": False, "message": f"续训事务异常已回滚：{exc}"}
        check = self.check_consistency()
        return {"ok": True,
                "message": f"续训完成，批次{batch['批次编号']}补写{len(appended)}个未完成周期",
                "batch": dict(batch), "appended": appended, "check": check}

    def list_batches(self, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        rows = [dict(b) for b in store.rows(BATCH_TABLE)]
        for field, expected in (filters or {}).items():
            if expected:
                rows = [b for b in rows if expected.casefold() in str(b.get(field, "")).casefold()]
        return sorted(rows, key=lambda b: (str(b.get("模型编号", "")), str(b.get("业务周期", "")),
                                           int(b.get("id", 0))))

    def list_results(self, 模型编号: str | None = None, 批次编号: str | None = None,
                     消费场景: str | None = None) -> list[dict[str, Any]]:
        """预测结果表：默认每个模型每个业务周期只定位最新批次；消费场景=对比时展示并存批次。"""
        batches = {str(b.get("批次编号", "")): b for b in store.rows(BATCH_TABLE)}
        rows = [dict(r) for r in store.rows(RESULT_TABLE) if str(r.get("批次编号", "")) in batches]
        rows = [r for r in rows if batches[str(r.get("批次编号", ""))].get("status") != "已归档"]
        if 消费场景 != "对比":
            latest: dict[tuple[str, str], int] = {}
            for b in store.rows(BATCH_TABLE):
                if b.get("status") in BATCH_ACTIVE_STATUSES:
                    key = (str(b.get("模型编号", "")), str(b.get("业务周期", "")))
                    latest[key] = max(latest.get(key, 0), int(b.get("id", 0)))
            keep = {str(b.get("批次编号", "")) for b in store.rows(BATCH_TABLE)
                    if int(b.get("id", 0)) in latest.values()}
            rows = [r for r in rows if str(r.get("批次编号", "")) in keep]
        if 模型编号:
            rows = [r for r in rows if 模型编号.casefold() in str(r.get("模型编号", "")).casefold()]
        if 批次编号:
            rows = [r for r in rows if 批次编号.casefold() in str(r.get("批次编号", "")).casefold()]
        for r in rows:
            batch = batches[str(r.get("批次编号", ""))]
            r["批次状态"] = batch.get("status", "")
        return sorted(rows, key=lambda r: (str(r.get("模型编号", "")), str(r.get("业务周期", "")),
                                           str(r.get("批次编号", "")), int(r.get("预测期次", 0))))

    def export_rows(self) -> list[dict[str, Any]]:
        """全份预测结果表：接收方按 V006 获取，含当前表内全部预测点。"""
        batches = {str(b.get("批次编号", "")): b for b in store.rows(BATCH_TABLE)}
        rows = [dict(r) for r in store.rows(RESULT_TABLE) if str(r.get("批次编号", "")) in batches]
        for r in rows:
            r["批次状态"] = batches[str(r.get("批次编号", ""))].get("status", "")
        return sorted(rows, key=lambda r: (str(r.get("模型编号", "")), str(r.get("业务周期", "")),
                                           str(r.get("批次编号", "")), int(r.get("预测期次", 0))))

    def list_archives(self) -> list[dict[str, Any]]:
        return [dict(a) for a in store.rows(ARCHIVE_TABLE)]

    def check_consistency(self) -> dict[str, Any]:
        """一致性核对：预测表出现无来源残留、重复期次或双批次事务异常时，回滚清残留并留档。"""
        snap = self._snapshot()
        try:
            batches = store.rows(BATCH_TABLE)
            results = store.rows(RESULT_TABLE)
            known = {str(b.get("批次编号", "")) for b in batches}
            removed: list[dict[str, Any]] = []
            anomalies: list[str] = []
            orphans = [r for r in results if str(r.get("批次编号", "")) not in known]
            if orphans:
                removed.extend(orphans)
                anomalies.append(f"无来源残留{len(orphans)}行")
            seen: set[tuple[Any, Any]] = set()
            dups: list[dict[str, Any]] = []
            for r in results:
                if r in orphans:
                    continue
                key = (r.get("批次编号"), r.get("预测期次"))
                if key in seen:
                    dups.append(r)
                else:
                    seen.add(key)
            if dups:
                removed.extend(dups)
                anomalies.append(f"重复期次{len(dups)}行")
            active: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for b in batches:
                if b.get("status") == "已完成":
                    active.setdefault((str(b.get("模型编号", "")), str(b.get("业务周期", ""))), []).append(b)
            for (model_code, period), group in active.items():
                if len(group) <= 1:
                    continue
                group.sort(key=lambda b: int(b.get("id", 0)))
                latest = group[-1]
                if latest.get("冲突策略") == "coexist":
                    continue  # 双批并存为消费场景显式选择，不算异常
                for old in group[:-1]:
                    stale = [r for r in results
                             if r.get("批次编号") == old.get("批次编号") and r not in removed]
                    removed.extend(stale)
                    old["status"] = "已归档"
                    old["finished_at"] = _now()
                    self._archive("回滚清残留", model_code, str(old.get("批次编号", "")),
                                  f"业务周期{period}预测表展示两相关预测结果集，事务异常按V006回滚旧批次",
                                  {"batch": dict(old), "results": stale})
                    anomalies.append(f"双批次异常:{old.get('批次编号', '')}")
            if removed:
                drop_ids = {id(r) for r in removed}
                store.rows(RESULT_TABLE)[:] = [r for r in results if id(r) not in drop_ids]
                self._archive("回滚清残留", "", "",
                              f"一致性核对清理{len(removed)}行残留：{'；'.join(a for a in anomalies if not a.startswith('双批次')) or '批次状态修正'}",
                              {"removed": removed})
        except Exception as exc:
            self._restore(snap)
            return {"ok": False, "message": f"核对事务异常已回滚：{exc}", "异常数": 0, "清理行数": 0, "留档": []}
        archives = [a["留档编号"] for a in store.rows(ARCHIVE_TABLE)]
        return {"ok": True,
                "message": "一致性核对完成" + (f"，清理{len(removed)}行残留" if removed else "，未发现异常"),
                "异常数": len(anomalies), "清理行数": len(removed), "异常明细": anomalies, "留档": archives}
