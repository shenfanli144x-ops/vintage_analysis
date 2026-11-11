# %%
from functools import lru_cache

import pandas as pd
import numpy as np
from data_loader import tmp

# 本模块所有函数都只读 tmp。需要加工时先 copy 出所需列，绝不原地写回 tmp，
# 否则跑到哪个回调都会悄悄改变后续分析的口径（之前的 cummax 回写就是这样）。
_VINTAGE_COLS = ["ID", "open_month", "month_on_book", "window"]

# ── 缓存约定 ────────────────────────────────────────────────────────────────
# 四个分析函数都是纯函数，输入只有 due / min_window / mob 这几个离散值，
# 可选组合极少（vintage 3×61、roll 7、km 3×61），但每次调用都要扫 104 万行。
# 用 lru_cache 把结果留在进程内：切 table tab、来回拨滑块、切换逾期维度时直接命中。
#
# 两条硬性约束：
#   1. 缓存里存的是同一个对象引用，公开入口**必须**返回 .copy()，
#      否则某个回调原地改一列就会永久污染缓存（和 tmp 被 cummax 回写是同一类事故）。
#      实测 61×61 的 copy 只要 0.01ms，可以忽略。
#   2. 参数要先规范化（min_window 一律 int），否则 24 和 24.0 会占两份缓存。


# %%
# ─── Vintage table ───────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def _filter_vintage_table(due, min_window):
    # 复制一份工作副本再加工，tmp 保持原样（due 列在 tmp 里始终是未累计的逐月标记）
    work = tmp[_VINTAGE_COLS + [due]].copy()

    # Cumulative max per ID: once overdue, always counted as overdue from that point forward
    work[due] = (
        work.sort_values(by="month_on_book", ascending=True).groupby("ID")[due].cummax()
    )

    # Denominator: pivot of unique ID counts per (open_month, month_on_book)
    # 直接用 work 里的 open_month / month_on_book / window，不再回 credit_info 重算一遍
    denominator = (
        work.query(f"window>={min_window}")
        .groupby(["open_month", "month_on_book"])["ID"]
        .nunique()
        .unstack()
    )

    # Numerator: sum of overdue per (open_month, month_on_book)
    numerator = (
        work.query(f"window>={min_window}")
        .groupby(["open_month", "month_on_book"])
        .agg({due: "sum"})
        .unstack()
    )

    # Rate
    # 分母缺失 = 该 open_month 的账龄还没走到这个 month_on_book，属于不可观测区域，
    # 必须保留 NaN 让图表断线、表格显示 "-"。之前用 replace({np.nan: 0}) 把这些格子
    # 填成 0%，导致每条 vintage 曲线右端都伪造归零（实测 49.2% 的格子受影响）。
    # inf 则来自分母为 0（有分子无分母，理论上不应出现），一并转成 NaN 以便暴露问题。
    vintage_table = numerator.div(denominator).replace([np.inf, -np.inf], np.nan)

    # numerator 经 .agg({due: "sum"}).unstack() 后列是 MultiIndex ('due30', mob)，
    # 相除后 vintage_table 也继承了这个结构；元组做列名会导致 Dash 序列化失败
    # （TypeError: keys must be str...），所以统一拍平成一维 month_on_book 列。
    for _df in (vintage_table, numerator):
        _df.columns = [c[1] if isinstance(c, tuple) else c for c in _df.columns]
        _df.columns.name = "month_on_book"

    return vintage_table, numerator, denominator


def filter_vintage_table(due="due30", min_window=0):
    """Returns (vintage_table, numerator, denominator)，均为独立副本。"""
    vt, num, den = _filter_vintage_table(due, int(min_window))
    return vt.copy(), num.copy(), den.copy()


# %%
# ─── Backward Roll Rate ──────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def _back_roll_rate(mob):
    all_statuses = sorted(tmp["STATUS"].unique().tolist())
    col_statuses = [s for s in all_statuses if s not in ("C", "X")]
    row_statuses = all_statuses

    # mob == "ALL" 时直接引用 tmp；下面的 .assign / .rename / .merge 都返回新对象，
    # 不会原地改动 tmp。切勿在这里补 tmp1[col] = ... 这种写法——那会污染基线。
    tmp1 = tmp if mob in (None, "ALL") else tmp.query(f"month_on_book == {mob}")

    back_roll = (
        tmp1.assign(last_months_balance=lambda x: x.MONTHS_BALANCE - 1)
        .rename(columns={"STATUS": "status"})
        .merge(
            tmp[["ID", "MONTHS_BALANCE", "STATUS"]].rename(
                columns={"STATUS": "last_status"}
            ),
            left_on=["ID", "last_months_balance"],
            right_on=["ID", "MONTHS_BALANCE"],
        )
        .query("last_months_balance > open_month")
        # observed=True：STATUS 现在是 category，默认 observed=False 会把 8×8 的
        # 类别笛卡尔积全部展开（哪怕某组合一个人都没有），矩阵因此多出空行；
        # 显式 observed=True 只保留真实出现过的组合，与转 category 之前完全一致。
        .groupby(["status", "last_status"], observed=True)
        .agg({"ID": "nunique"})
        .unstack()
    )
    back_roll.columns = back_roll.columns.droplevel(0)
    back_roll = back_roll.fillna(0)

    back_roll = back_roll.loc[
        [s for s in row_statuses if s in back_roll.index],
        [s for s in col_statuses if s in back_roll.columns],
    ]
    # col_statuses 原来是全局状态列表，但某些 mob（如 24/60）里并非所有状态都出现过，
    # 矩阵只保留了实际存在的列；这里同步收窄，否则上层 col_totals[ls] 会 KeyError。
    col_statuses = [s for s in col_statuses if s in back_roll.columns]

    col_totals = back_roll.sum(axis=0)
    df_rates = back_roll.div(col_totals, axis=1).replace({np.nan: 0, np.inf: 0})
    df_rates = df_rates.reset_index()
    back_roll = back_roll.reset_index()

    return back_roll, df_rates, tuple(col_statuses), col_totals


def back_roll_rate(mob):
    """
    Returns (df_counts, df_rates, col_statuses, col_totals)
    df_counts: status x last_status pivot of unique ID counts
    df_rates:  same shape, values = cell / column total
    col_statuses: list of column statuses (no C, X)
    col_totals: Series indexed by last_status = column sums
    """
    df_counts, df_rates, col_statuses, col_totals = _back_roll_rate(mob)
    return df_counts.copy(), df_rates.copy(), list(col_statuses), col_totals.copy()


# %%
# ─── Observe Window CDF ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _observe_window_cdf():
    unique_ids = tmp.drop_duplicates(subset="ID")
    window_counts = unique_ids.window.value_counts().sort_index()
    cdf = window_counts.cumsum() / window_counts.sum()
    return cdf.index.tolist(), cdf.values.tolist()


def observe_window_cdf():
    """Returns (x, y) arrays for the window-length CDF.
    Each account (ID) contributes once; CDF of window length across unique IDs.

    只跟 ID 的观测长度有关，与逾期定义无关，因此不接受 due / min_window 参数。
    """
    x, y = _observe_window_cdf()
    return list(x), list(y)


# %%
# ─── Survival Analysis (Kaplan-Meier) ────────────────────────────────────────
@lru_cache(maxsize=64)
def _survival_curve(due, min_window):
    # Filter by window length
    # 布尔索引本身就会返回副本，后续 .assign 也是新对象，tmp 不会被改动，
    # 所以这里不需要再 .copy() 一次（1M 行再复制一遍纯属浪费内存）。
    # 注意 due 列是 tmp 里的原始逐月标记（未经 cummax）：事件月取
    # min(month_on_book | due==1)、删失取 max(month_on_book)，这两个统计量
    # 对 cummax 与否都不变，因此结果与 filter_vintage_table 是否先跑过无关。
    filtered = tmp[tmp["window"] > min_window]

    # min_window 拉满（>= MAX_WINDOW）时筛选结果为空，lifelines 会抛
    # "Values must be numeric"；这里直接返回空曲线，让图空着而不是 500。
    if filtered.empty:
        return (), ()

    event = filtered.groupby("ID", as_index=False).agg(
        {"month_on_book": "max", due: "max"}
    )

    km = pd.concat(
        [
            event.query(f"{due}==0"),
            filtered.assign(is_event=lambda x: x.groupby("ID")[due].transform("max"))
            .query(f"is_event==1 and {due}==1")
            .groupby("ID", as_index=False)
            .agg({"month_on_book": "min", due: "max"}),
        ]
    )

    if km.empty:
        return (), ()

    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(km["month_on_book"], event_observed=km[due], label=due)

    timeline = tuple(sorted(kmf.survival_function_.index.tolist()))
    surv_probs = tuple(kmf.survival_function_[due].values.tolist())
    return timeline, surv_probs


def survival_curve(due="due30", min_window=0):
    """
    Returns (timeline, survival_probs) for Kaplan-Meier survival curve.
    Builds event table filtered by min_window.

    口径：**非 cummax**。直接使用 tmp 里逐月的原始 due 标记，不做 ever-overdue 累计。
      - 事件时点 = 该客户第一次出现 due==1 的 month_on_book
      - 删失时点 = 该客户最后观测到的 month_on_book
      - 事件标记 = 该客户是否曾出现过 due==1
    这三个统计量在 cummax 前后完全等价（cummax 只在首次出现 1 之后把 0 改写成 1，
    不影响首次出现的位置，也不影响 max），所以本曲线与 filter_vintage_table 是否先跑过无关，
    也不会被 vintage 的 "ever overdue" 口径污染。
    """
    timeline, surv_probs = _survival_curve(due, int(min_window))
    return list(timeline), list(surv_probs)


def cache_info():
    """调试用：查看四个缓存的命中情况。"""
    return {
        "vintage": _filter_vintage_table.cache_info(),
        "roll": _back_roll_rate.cache_info(),
        "cdf": _observe_window_cdf.cache_info(),
        "km": _survival_curve.cache_info(),
    }
