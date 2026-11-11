"""app.py 回调回归测试。

用法：
    python3 -m pytest test_app.py -q

覆盖 4 个图表的回调（含边界参数）、缓存副本隔离、以及「不可观测格子必须是 NaN」这条
最容易复发的约定。改 app.py / analysis.py 前后各跑一次，比人眼看图可靠。
"""

import json

import numpy as np
import pytest

import analysis as an
import app as appmod

DUE_ALL = ["due00", "due30", "due60"]
TABS = ["rate", "numerator", "denominator"]


@pytest.fixture(scope="session")
def client():
    appmod.app._setup_server()
    return appmod.app.server.test_client()


def post(client, payload):
    r = client.post("/_dash-update-component", json=payload)
    assert (
        r.status_code == 200
    ), f"{payload.get('output')} -> {r.status_code}: {r.data[:300]}"
    return r.get_json()["response"]


def vintage_chart(client, due, mw):
    return post(
        client,
        {
            "output": "vintage-chart.figure",
            "inputs": [
                {"id": "due-select", "property": "value", "value": due},
                {"id": "window-slider", "property": "value", "value": mw},
            ],
            "changedPropIds": [],
        },
    )["vintage-chart"]["figure"]


def vintage_table(client, due, mw, tab, compact):
    resp = post(
        client,
        {
            "output": "..vintage-table.data...vintage-table.columns..",
            "outputs": [
                {"id": "vintage-table", "property": "data"},
                {"id": "vintage-table", "property": "columns"},
            ],
            "inputs": [
                {"id": "due-select", "property": "value", "value": due},
                {"id": "window-slider", "property": "value", "value": mw},
                {"id": "vintage-table-tabs", "property": "value", "value": tab},
                {"id": "table-compact", "property": "value", "value": compact},
            ],
            "changedPropIds": [],
        },
    )
    return resp["vintage-table"]["data"], resp["vintage-table"]["columns"]


def km_chart(client, due, mw):
    resp = post(
        client,
        {
            "output": "..survival-chart.figure...survival-desc.children..",
            "outputs": [
                {"id": "survival-chart", "property": "figure"},
                {"id": "survival-desc", "property": "children"},
            ],
            "inputs": [
                {"id": "survival-due-tabs", "property": "value", "value": due},
                {"id": "window-slider", "property": "value", "value": mw},
            ],
            "changedPropIds": [],
        },
    )
    return resp["survival-chart"]["figure"], resp["survival-desc"]["children"]


# ── Vintage 图 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("due", DUE_ALL)
@pytest.mark.parametrize("mw", [0, 24, 55, 60])
def test_vintage_chart(client, due, mw):
    fig = vintage_chart(client, due, mw)
    vt, _, _ = an.filter_vintage_table(due, mw)

    lines = [t for t in fig["data"] if t["x"] and t["x"][0] is not None]
    assert len(lines) == len(vt.index), "每条 cohort 一条线"

    # 每条线都上图例（61 项），且不应再有 colorbar 占位 trace
    assert not any(
        t.get("marker", {}).get("showscale") for t in fig["data"]
    ), "不应再有 colorbar trace"
    assert all(
        t.get("showlegend", True) is not False for t in lines
    ), "每条线都要上图例"
    names = [t["name"] for t in lines]
    assert len(set(names)) == len(names), "图例名不能重复"

    order = sorted(vt.index.tolist(), reverse=True)
    for trace, om in zip(lines, order):
        row = vt.loc[om]
        assert trace["name"] == f"open_{om}"
        assert len(trace["x"]) == len(row)
        for got, col in zip(trace["y"], vt.columns):
            expected = row[col]
            if np.isnan(expected):
                assert got is None, f"open_{om} mob={col} 不可观测格子应为 None（断线）"
            else:
                assert got == pytest.approx(expected * 100)


def test_vintage_chart_has_no_fake_zero(client):
    """回归：不可观测区域曾经被 replace({nan:0}) 填成 0%，曲线右端会假性归零。"""
    fig = vintage_chart(client, "due30", 0)
    newest = [t for t in fig["data"] if t.get("name") == "open_0"][0]
    assert newest["y"] == [pytest.approx(newest["y"][0])] + [None] * 60


# ── Vintage 表 ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("due", DUE_ALL)
@pytest.mark.parametrize("tab", TABS)
def test_vintage_table(client, due, tab):
    data, columns = vintage_table(client, due, 0, tab, [])
    assert data and columns
    assert columns[0]["id"] == "open_month"
    ids = {c["id"] for c in columns}
    for row in data[:3]:
        assert set(row) == ids, "data 的 key 必须与 columns 的 id 一一对应"
    # 每行长度 = 表头数，且 rate 列都带 % 或 "-"（首列 open_month 是整数）
    if tab == "rate":
        for row in data:
            for col in columns[1:]:
                assert row[col["id"]] == "-" or str(row[col["id"]]).endswith("%")


@pytest.mark.parametrize("due", DUE_ALL)
def test_vintage_table_compact(client, due):
    full, _ = vintage_table(client, due, 0, "rate", [])
    compact, cols = vintage_table(client, due, 0, "rate", ["on"])
    assert len(cols) < len(full[0]), "紧凑视图应裁掉低频列"
    assert cols[0]["id"] == "open_month"


# ── Roll rate ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mob", ["ALL", 5, 6, 12, 24, 36, 48, 60])
def test_rollrate(client, mob):
    fig = post(
        client,
        {
            "output": "rollrate-chart.figure",
            "inputs": [{"id": "mob-select", "property": "value", "value": mob}],
            "changedPropIds": [],
        },
    )["rollrate-chart"]["figure"]
    heat = [t for t in fig["data"] if t["type"] == "heatmap"]
    assert len(heat) == 1
    assert heat[0]["z"], "热力图不应为空"
    # 每行长度与列一致，否则 plotly 会画错位
    assert all(len(row) == len(heat[0]["x"]) for row in heat[0]["z"])


# ── Window CDF ───────────────────────────────────────────────────────────────
def test_cdf(client):
    """CDF 只跟账户观测长度有关，与 due / min_window 无关（入参是占位用的 id）。"""
    fig = post(
        client,
        {
            "output": "observe-chart.figure",
            "inputs": [{"id": "due-select", "property": "id", "value": "due-select"}],
            "changedPropIds": [],
        },
    )["observe-chart"]["figure"]
    x, y = fig["data"][0]["x"], fig["data"][0]["y"]
    assert len(x) == len(y)
    assert np.all(np.diff(y) >= 0), "CDF 必须单调不减"
    assert y[-1] == pytest.approx(1.0)


def test_every_callback_has_input():
    """Dash 2.x 不允许无 Input 的回调：前端会抛 "A callback is missing Inputs"，
    后果是**整页所有回调都不执行**（不是只有那一个图空着）。这条守住别再犯。"""
    for key, spec in appmod.app.callback_map.items():
        assert spec.get("inputs"), f"{key} 没有 Input —— 整页回调都会失效"


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("due", DUE_ALL)
@pytest.mark.parametrize("mw", [0, 24, 55, 60])
def test_km(client, due, mw):
    fig, desc = km_chart(client, due, mw)
    assert desc, "维度说明文案不应为空"
    t, s = an.survival_curve(due, mw)

    if not t:
        # min_window 拉到最大值时样本为空，应显示占位说明而不是 500
        assert not fig["data"]
        assert fig["layout"]["annotations"], "空样本应给出占位提示"
        return

    trace = fig["data"][0]
    assert list(trace["x"]) == list(t)
    assert np.allclose(np.array(trace["y"]), np.array(s))
    assert all(
        a >= b - 1e-12 for a, b in zip(trace["y"], trace["y"][1:])
    ), "生存曲线不应上升"


def test_km_tabs_are_independent(client):
    """KM 的维度由卡片内选项卡控制，三个维度必须给出不同曲线。"""
    ends = {due: km_chart(client, due, 0)[0]["data"][0]["y"][-1] for due in DUE_ALL}
    assert ends["due00"] < ends["due30"] < ends["due60"]


# ── 缓存与副本隔离 ───────────────────────────────────────────────────────────
def test_cached_results_are_copies():
    """缓存必须返回副本：调用方原地改一列不能污染下一次调用。"""
    vt1, num1, _ = an.filter_vintage_table("due60", 12)
    vt1.iloc[:, :] = -999
    num1.iloc[:, :] = -999
    vt2, num2, _ = an.filter_vintage_table("due60", 12)
    assert not (vt2 == -999).all().all(), "缓存被调用方污染了"
    assert not (num2 == -999).all().all()


def test_min_window_int_normalization():
    """24 和 24.0 必须命中同一份缓存，否则滑块拖动会不断产生新条目。"""
    before = an._filter_vintage_table.cache_info()
    an.filter_vintage_table("due30", 24)
    an.filter_vintage_table("due30", 24.0)
    after = an._filter_vintage_table.cache_info()
    assert after.misses - before.misses <= 1


def test_tmp_is_not_mutated():
    """tmp 是只读基线：调用一圈分析函数后必须逐值不变。"""
    import data_loader as dl

    before = dl.tmp.copy()
    for due in DUE_ALL:
        for mw in (0, 24, 60):
            an.filter_vintage_table(due, mw)
    an.survival_curve("due60", 12)
    an.back_roll_rate(24)
    an.observe_window_cdf()
    assert before.equals(dl.tmp), "tmp 被原地修改了"
    assert sorted(dl.tmp.due30.unique().tolist()) == [
        0,
        1,
    ], "due 应保持未 cummax 的逐月标记"
