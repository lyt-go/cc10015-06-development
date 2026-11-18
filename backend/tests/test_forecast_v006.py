"""V006 预测批次与预测结果表链路测试。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.store import store
from app.services.forecast_model import ForecastModelService
from app.services.metric_def import MetricDefService

HIST = [0.10, 0.12, 0.11, 0.13, 0.15, 0.14, 0.16]


def _clear_all():
    for k in list(store._tables.keys()):
        store._tables[k] = []


def _model(fc, **over):
    values = {"模型编号": "FC-001", "预测对象": "点击率", "历史窗口": "7", "预测周期": "3",
              "模型类型": "moving_average", "参数配置": "window=3", "训练批次": "", "评估误差": ""}
    values.update(over)
    entry, ms = fc.create_entry(values)
    assert entry is not None, ms
    return entry


def test_train_persists_batch_and_results_atomically():
    """训练批次与预测结果同时落库，预测点绑定训练批次与指标口径，刷新结果稳定。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)
    result = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    assert result["ok"], result
    batch = result["batch"]
    batches = store.rows("forecast_batch")
    results = store.rows("forecast_result")
    assert len(batches) == 1 and batches[0]["批次编号"] == batch["批次编号"]
    assert batches[0]["status"] == "已完成"
    assert len(results) == 3, "预测周期=3 应写入3个预测点"
    for row in results:
        assert row["批次编号"] == batch["批次编号"], "预测点必须绑定训练批次"
        assert row["指标编号"] and row["口径版本"] and "精度位数" in row, "预测点必须绑定指标口径"
        assert row["来源版本"] == "V006"
    assert entry["训练批次"] == batch["批次编号"], "模型应定位到唯一训练批次"
    first = fc.list_results(模型编号="FC-001")
    second = fc.list_results(模型编号="FC-001")
    assert first == second and len(first) == 3, "刷新不应出现另一组无来源数值"


def test_threshold_rejects_out_of_range_and_rolls_back():
    """超出范围或越过上限的预测点按阈值拒绝；整批被拒时回滚并留档。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc, 模型类型="linear", 参数配置="")
    history = [0.1, 0.2, 0.3, 0.4]
    # 线性外推 0.5/0.6/0.7，上限0.55：第1点通过，后2点越限拒绝
    result = fc.train(entry["id"], {"历史数据": history, "业务周期": "2026-11", "阈值上限": "0.55"})
    assert result["ok"], result
    assert len(store.rows("forecast_result")) == 1
    assert result["batch"]["拒绝点数"] == 2
    assert all("越过阈值上限" in r["拒绝原因"] for r in result["rejected"])
    # 上限0.4：全部越限，整批回滚不落库并留档
    before_batches = len(store.rows("forecast_batch"))
    before_results = len(store.rows("forecast_result"))
    result2 = fc.train(entry["id"], {"历史数据": history, "业务周期": "2026-12", "阈值上限": "0.4"})
    assert not result2["ok"]
    assert len(store.rows("forecast_batch")) == before_batches, "整批拒绝不得留下批次"
    assert len(store.rows("forecast_result")) == before_results, "整批拒绝不得留下结果行"
    archives = store.rows("forecast_archive")
    assert any(a["类型"] == "整批拒绝" for a in archives), "整批拒绝应按V006留档"


def test_repeat_train_same_period_replace_archives_old():
    """同业务周期重复训练：最新批次替换当前结果，旧批次归档，预测表只显示最新批次。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)
    r1 = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    assert r1["ok"]
    batch1 = r1["batch"]["批次编号"]
    r2 = fc.train(entry["id"], {"历史数据": [0.2] * 7, "业务周期": "2026-11"})
    assert r2["ok"], r2
    batch2 = r2["batch"]["批次编号"]
    batches = {b["批次编号"]: b for b in store.rows("forecast_batch")}
    assert batches[batch1]["status"] == "已归档", "旧批次应归档"
    assert batches[batch2]["status"] == "已完成"
    rows = fc.list_results(模型编号="FC-001")
    assert rows and {r["批次编号"] for r in rows} == {batch2}, "预测表只显示最新批次"
    assert all(r["预测值"] == 0.2 for r in rows)
    assert any(a["类型"] == "批次归档" and a["批次编号"] == batch1
               for a in store.rows("forecast_archive")), "旧批次归档应按V006留档"
    assert entry["训练批次"] == batch2, "重复训练后仍定位唯一训练批次"


def test_coexist_policy_keeps_both_batches_for_compare():
    """双批并存：预测表默认显示最新批次，消费场景=对比时展示两个批次及各自数据。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)
    r1 = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    batch1 = r1["batch"]["批次编号"]
    r2 = fc.train(entry["id"], {"历史数据": [0.2] * 7, "业务周期": "2026-11", "冲突策略": "coexist"})
    assert r2["ok"], r2
    batch2 = r2["batch"]["批次编号"]
    batches = store.rows("forecast_batch")
    assert sum(1 for b in batches if b["status"] == "已完成") == 2, "并存策略下两个批次都保留"
    latest_only = fc.list_results(模型编号="FC-001")
    assert {r["批次编号"] for r in latest_only} == {batch2}, "默认消费场景只定位最新批次"
    compared = fc.list_results(模型编号="FC-001", 消费场景="对比")
    assert {r["批次编号"] for r in compared} == {batch1, batch2}, "对比场景展示两个批次各自数据"


def test_interrupt_then_resume_only_appends_missing_periods():
    """训练中断后只能补写未完成周期，不能覆盖上一完整预测批次。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)
    complete = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-10"})
    assert complete["ok"]
    complete_rows_before = [dict(r) for r in fc.list_results(模型编号="FC-001", 消费场景="对比")
                            if r["批次编号"] == complete["batch"]["批次编号"]]
    # 同模型新周期训练中断在第1期
    interrupted = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11", "模拟中断期次": "1"})
    assert interrupted["ok"], interrupted
    assert interrupted["batch"]["status"] == "已中断"
    assert interrupted["batch"]["已完成期次"] == 1
    # 中断批次存在时不允许重新训练，只能续训
    again = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    assert not again["ok"] and "续训" in again["message"]
    rows_before = list(store.rows("forecast_result"))
    resumed = fc.resume(entry["id"])
    assert resumed["ok"], resumed
    assert resumed["batch"]["status"] == "已完成"
    assert resumed["batch"]["已完成期次"] == resumed["batch"]["预测周期数"]
    rows_after = list(store.rows("forecast_result"))
    assert len(rows_after) == len(rows_before) + 2, "续训只补写剩余2个周期"
    assert all(r in rows_after for r in rows_before), "已有预测点不得被改写"
    complete_rows_after = [dict(r) for r in fc.list_results(模型编号="FC-001", 消费场景="对比")
                           if r["批次编号"] == complete["batch"]["批次编号"]]
    assert complete_rows_before == complete_rows_after, "上一完整预测批次不得被覆盖"


def test_consistency_check_rolls_back_residue_and_double_batch():
    """预测表展示两相关预测结果集等事务异常：回滚清残留并按V006留档核对。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)
    r1 = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    batch1 = r1["batch"]["批次编号"]
    # 制造无来源残留与重复期次
    store.rows("forecast_result").append({
        "id": 9001, "批次编号": "FB-9999", "模型编号": "FC-001", "指标编号": "点击率",
        "口径版本": "v1", "精度位数": 4, "业务周期": "2026-11", "预测期次": 1,
        "预测值": 9.99, "校验状态": "通过", "来源版本": "V006", "created_at": "2026-11-18 00:00:00"})
    store.rows("forecast_result").append(dict(store.rows("forecast_result")[0], id=9002))
    # 制造双批次异常：并存完成的两个批次且最新批次策略为替换
    r2 = fc.train(entry["id"], {"历史数据": [0.2] * 7, "业务周期": "2026-11", "冲突策略": "coexist"})
    batch2 = r2["batch"]["批次编号"]
    for b in store.rows("forecast_batch"):
        if b["批次编号"] == batch2:
            b["冲突策略"] = "replace"
    check = fc.check_consistency()
    assert check["ok"], check
    assert check["清理行数"] >= 2, "无来源残留与重复期次都应清理"
    codes = {r["批次编号"] for r in store.rows("forecast_result")}
    assert "FB-9999" not in codes, "无来源残留应清除"
    batches = {b["批次编号"]: b for b in store.rows("forecast_batch")}
    assert batches[batch1]["status"] == "已归档", "双批次异常应回滚旧批次"
    assert batches[batch2]["status"] == "已完成"
    shown = {r["批次编号"] for r in fc.list_results(模型编号="FC-001")}
    assert shown == {batch2}, "清理后预测表只保留唯一来源"
    archives = store.rows("forecast_archive")
    assert any(a["类型"] == "回滚清残留" for a in archives), "回滚清残留应按V006留档"


def test_train_transaction_failure_rolls_back(monkeypatch):
    """训练事务异常：批次与结果同时回滚，不留半提交状态。"""
    _clear_all()
    fc = ForecastModelService()
    entry = _model(fc)

    def boom(self, entry, values):
        raise RuntimeError("模拟落库失败")

    monkeypatch.setattr(ForecastModelService, "_predict_values", boom)
    result = fc.train(entry["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    assert not result["ok"] and "回滚" in result["message"]
    assert store.rows("forecast_batch") == [], "异常时批次不得落库"
    assert store.rows("forecast_result") == [], "异常时结果不得落库"
    assert any(a["类型"] == "回滚清残留" for a in store.rows("forecast_archive"))


def test_metric_def_reads_forecast_result_and_validates_precision():
    """跨模块联动（V006）：指标定义读取预测结果表的指标标识和精度位数，校验展示精度。"""
    _clear_all()
    md = MetricDefService()
    entry, ms = md.create_entry({"指标编号": "M-100", "指标名称": "点击率", "计算公式": "c/i",
                                 "统计粒度": "日", "时间窗口": "7", "基准取值": "0.1",
                                 "精度位数": "2", "口径版本": "v2"})
    assert entry is not None, ms
    fc = ForecastModelService()
    model = _model(fc, 预测对象="M-100")
    trained = fc.train(model["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    assert trained["ok"], trained
    rows = store.rows("forecast_result")
    assert all(r["精度位数"] == 2 and r["口径版本"] == "v2" for r in rows), "预测点应绑定指标口径"
    report = md.validate_forecast_precision()
    assert report["total"] == 1
    item = report["items"][0]
    assert item["指标编号"] == "M-100" and item["校验结论"] == "通过", item
    assert item["结果表精度"] == [2] and item["定义精度"] == 2
    # 塞入一个超精度残留值后应被指标定义模块检出
    store.rows("forecast_result").append(dict(rows[0], id=9001, 预测值=0.12345))
    report2 = md.validate_forecast_precision()
    assert report2["items"][0]["精度超限点"] == 1
    assert "超过精度位数" in report2["items"][0]["校验结论"]


def test_workbench_and_detail_share_same_results():
    """工作台面板与详情页保留同一结果；接收方可获取全份预测结果表。"""
    _clear_all()
    fc = ForecastModelService()
    e1 = _model(fc, 模型编号="FC-001")
    e2 = _model(fc, 模型编号="FC-002")
    fc.train(e1["id"], {"历史数据": HIST, "业务周期": "2026-11"})
    fc.train(e2["id"], {"历史数据": [0.3] * 7, "业务周期": "2026-11"})
    panel = fc.list_results(模型编号="FC-001")
    detail = fc.get_entry_detail(e1["id"])
    assert detail is not None
    assert detail["results"] == panel, "详情页与工作台面板必须同源"
    full = fc.export_rows()
    assert len(full) == len(store.rows("forecast_result")) == 6, "接收方应获取全份预测结果表"
    assert {r["模型编号"] for r in full} == {"FC-001", "FC-002"}
