# %%
import os

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, callback

from data_loader import MAX_WINDOW, ALL_MOBS
from analysis import (
    filter_vintage_table,
    back_roll_rate,
    observe_window_cdf,
    survival_curve,
)

# %%
# ─── Dash app ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Credit Analysis"

# ── 设计变量：颜色 / 字号只在这里定义一次 ────────────────────────────────────
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#898781"
BORDER = "#c3c2b7"
GRID = "#e1e0d9"
PAPER = "#fcfcfb"

# 逾期定义维度：KM 卡片的选项卡，颜色固定，便于跨维度对照
DUE_OPTIONS = [
    ("due00", "Due00"),
    ("due30", "Due30"),
    ("due60", "Due60"),
]
DUE_COLORS = {"due00": "#2a78d6", "due30": "#eb6834", "due60": "#6250d6"}
DUE_DESC = {
    "due00": "当前月有逾期即算事件（any delinquency）",
    "due30": "逾期 30 天及以上才算事件（M1+）",
    "due60": "逾期 60 天及以上才算事件（M2+）",
}

H1_STYLE = {"fontSize": 22, "fontWeight": 600, "margin": "0 0 20px 0"}
H2_STYLE = {"fontSize": 18, "fontWeight": 600, "margin": "0 0 12px 0"}
CAPTION_STYLE = {"fontSize": 12, "color": MUTED, "margin": "0 0 10px 2px"}

CARD_STYLE = {
    "padding": "20px",
    "background": PAPER,
    "border": "1px solid rgba(11,11,11,0.10)",
    "borderRadius": 8,
}

FILTER_STYLE = {
    "display": "flex",
    "gap": "32px",
    "alignItems": "center",
    "flexWrap": "wrap",
    "marginBottom": "20px",
    "padding": "16px 20px",
    "background": PAPER,
    "border": "1px solid rgba(11,11,11,0.10)",
    "borderRadius": 8,
}

# 紧凑版选项卡样式（放在卡片标题右侧）
TAB_STYLE = {
    "padding": "5px 12px",
    "fontSize": 12,
    "borderTop": "2px solid transparent",
    "borderBottom": f"1px solid {GRID}",
    "color": MUTED,
    "background": "transparent",
}
TAB_SELECTED_STYLE = {
    "padding": "5px 12px",
    "fontSize": 12,
    "borderTop": "2px solid #2a78d6",
    "borderBottom": f"1px solid {PAPER}",
    "color": INK,
    "fontWeight": 600,
    "background": "transparent",
}

# Vintage Table 的选项卡：Dash 默认会把 Tabs 拉满整行、三等分，
# 文字挤在左边一片空白里很难看。这里改成宽度由文字决定，左右留等量内边距。
VTAB_STYLE = {
    **TAB_STYLE,
    "padding": "7px 18px",
    "fontSize": 13,
    "whiteSpace": "nowrap",
    "width": "auto",
    "flex": "0 0 auto",
}
VTAB_SELECTED_STYLE = {
    **TAB_SELECTED_STYLE,
    "padding": "7px 18px",
    "fontSize": 13,
    "whiteSpace": "nowrap",
    "width": "auto",
    "flex": "0 0 auto",
}


# %%
# ─── 图表样式工具 ────────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color, alpha):
    """#rrggbb -> rgba(...)，用于填充色跟随主色。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# vintage 曲线配色：8 色循环。
# 说明：用过一段连续 Blues 色带 + colorbar 的方案，色带本身没问题，但 colorbar 只能
# 表达「新旧」这个连续量，读不出具体是哪个 cohort。这里改回逐条图例，按名字点名。
SERIES_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]

_AXIS_BASE = dict(
    title_font=dict(size=12, color=MUTED),
    linecolor=BORDER,
    tickcolor=BORDER,
    tickfont=dict(size=11, color=FAINT),
    gridcolor=GRID,
    gridwidth=1,
)


def _base_layout(
    fig, x_title="", y_title="", xaxis=None, yaxis=None, legend=None, margin=None
):
    """四个图共用的底板：透明背景、统一轴样式、统一 hover 样式。

    以前这段在每个回调里逐字重复 4 遍，改一个色值要改 4 处。
    """
    xax = {**_AXIS_BASE, "title": x_title, **(xaxis or {})}
    yax = {**_AXIS_BASE, "title": y_title, **(yaxis or {})}
    layout = dict(
        margin=margin or dict(l=52, r=24, t=40, b=56),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=xax,
        yaxis=yax,
        hoverlabel=dict(
            bgcolor=PAPER,
            bordercolor="rgba(11,11,11,0.10)",
            font=dict(size=13, color=INK),
        ),
    )
    if legend is not None:
        layout["legend"] = legend
    fig.update_layout(**layout)
    return fig


def _empty_figure(message):
    """筛选后样本为空时的占位图，避免直接 500。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=13, color=MUTED),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        margin=dict(l=52, r=24, t=40, b=56),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _loading(graph):
    """慢回调的加载态：拖滑块时先转圈，而不是让人以为没点上。"""
    return dcc.Loading(
        graph,
        type="circle",
        color="#2a78d6",
        parent_style={"minHeight": "0"},
        style={"minHeight": "0"},
    )


# %%
# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        # ── Section 1: Vintage Analysis ──────────────────────────────────────────
        html.H1("1. Vintage Analysis", style=H1_STYLE),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Due:", style={"marginRight": 8, "fontSize": 14}),
                        dcc.Dropdown(
                            id="due-select",
                            options=[
                                {"label": label.lower(), "value": value}
                                for value, label in DUE_OPTIONS
                            ],
                            value="due30",
                            clearable=False,
                            style={"width": 160},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": 8},
                ),
                html.Div(
                    [
                        html.Label(
                            "min_window:", style={"marginRight": 8, "fontSize": 14}
                        ),
                        dcc.Slider(
                            id="window-slider",
                            min=0,
                            max=MAX_WINDOW,
                            step=1,
                            value=0,
                            marks={
                                i: str(i)
                                for i in range(
                                    0, MAX_WINDOW + 1, max(1, MAX_WINDOW // 6)
                                )
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"flex": 1, "minWidth": 200, "maxWidth": 500},
                ),
            ],
            style=FILTER_STYLE,
        ),
        html.Div(
            [
                html.H2("Vintage Curve", style=H2_STYLE),
                # 高度跟着图例走：底部 176px 留给 61 项图例，绘图区才有 300+px
                _loading(dcc.Graph(id="vintage-chart", style={"height": 520})),
            ],
            style=CARD_STYLE,
        ),
        html.Div(
            [
                html.H2("Vintage Table", style=H2_STYLE),
                html.Div(
                    [
                        dcc.Tabs(
                            id="vintage-table-tabs",
                            value="rate",
                            style={"width": "auto", "flex": "0 1 auto"},
                            children=[
                                dcc.Tab(
                                    label="%",
                                    value="rate",
                                    style=VTAB_STYLE,
                                    selected_style=VTAB_SELECTED_STYLE,
                                ),
                                dcc.Tab(
                                    label="Numerator",
                                    value="numerator",
                                    style=VTAB_STYLE,
                                    selected_style=VTAB_SELECTED_STYLE,
                                ),
                                dcc.Tab(
                                    label="Denominator",
                                    value="denominator",
                                    style=VTAB_STYLE,
                                    selected_style=VTAB_SELECTED_STYLE,
                                ),
                            ],
                        ),
                        dcc.Checklist(
                            id="table-compact",
                            options=[
                                {
                                    "label": "紧凑视图（隐藏不足半数 cohort 可观测的月份）",
                                    "value": "on",
                                }
                            ],
                            value=[],
                            style={"fontSize": 12, "color": MUTED},
                            inputStyle={"marginRight": 6},
                            labelStyle={
                                "display": "flex",
                                "alignItems": "center",
                                "cursor": "pointer",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "gap": 16,
                        "flexWrap": "wrap",
                        "marginBottom": 12,
                    },
                ),
                dash_table.DataTable(
                    id="vintage-table",
                    style_table={"overflowX": "auto", "minWidth": "100%"},
                    style_cell={
                        "textAlign": "right",
                        "padding": "7px 10px",
                        "fontSize": 13,
                        "fontVariantNumeric": "tabular-nums",
                        "borderBottom": f"1px solid {GRID}",
                        "minWidth": "52px",
                        "width": "52px",
                        "maxWidth": "72px",
                    },
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "open_month"},
                            "textAlign": "left",
                            "width": "96px",
                            "minWidth": "96px",
                            "fontWeight": 500,
                        },
                    ],
                    style_header={
                        "fontWeight": 500,
                        "color": MUTED,
                        "position": "sticky",
                        "top": 0,
                        "background": PAPER,
                        "borderBottom": f"1px solid {GRID}",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f7"},
                    ],
                    sort_action="native",
                    fixed_columns={"data": 1},
                    fill_width=False,
                ),
            ],
            style={**CARD_STYLE, "marginBottom": 40},
        ),
        # ── Section 2: Backward Roll Rate ────────────────────────────────────────
        html.H1("2. Backward Roll Rate", style=H1_STYLE),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "month_on_book:", style={"marginRight": 8, "fontSize": 14}
                        ),
                        dcc.Dropdown(
                            id="mob-select",
                            options=(
                                [{"label": "ALL", "value": "ALL"}]
                                + [
                                    {"label": str(m), "value": m}
                                    for m in ALL_MOBS
                                    if m >= 5
                                ]
                            ),
                            value="ALL",
                            clearable=False,
                            style={"width": 160},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": 8},
                ),
            ],
            style=FILTER_STYLE,
        ),
        html.Div(
            [
                html.H2("Roll Rate Heatmap", style=H2_STYLE),
                _loading(dcc.Graph(id="rollrate-chart", style={"height": 520})),
            ],
            style=CARD_STYLE,
        ),
        # ── Section 3 & 4: side-by-side ──────────────────────────────────────────
        html.H1("3 & 4. Observe Window & Survival Analysis", style=H1_STYLE),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Window CDF", style=H2_STYLE),
                        html.Div(
                            "每个账户贡献一次；只与观测长度有关，与逾期定义无关",
                            style=CAPTION_STYLE,
                        ),
                        _loading(dcc.Graph(id="observe-chart", style={"height": 400})),
                    ],
                    style={"flex": 1, "minWidth": 0},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2(
                                    "Kaplan-Meier Survival Curve",
                                    style={
                                        "fontSize": 18,
                                        "fontWeight": 600,
                                        "margin": 0,
                                    },
                                ),
                                dcc.Tabs(
                                    id="survival-due-tabs",
                                    value="due30",
                                    children=[
                                        dcc.Tab(
                                            label=label,
                                            value=value,
                                            style=TAB_STYLE,
                                            selected_style=TAB_SELECTED_STYLE,
                                        )
                                        for value, label in DUE_OPTIONS
                                    ],
                                    style={"height": 32, "minWidth": 250},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "space-between",
                                "gap": 16,
                                "marginBottom": 12,
                            },
                        ),
                        html.Div(
                            DUE_DESC["due30"], id="survival-desc", style=CAPTION_STYLE
                        ),
                        _loading(dcc.Graph(id="survival-chart", style={"height": 400})),
                    ],
                    style={"flex": 1, "minWidth": 0},
                ),
            ],
            style={**CARD_STYLE, "display": "flex", "gap": "24px", "flexWrap": "wrap"},
        ),
    ],
    style={
        "padding": "24px",
        "fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif",
    },
)


# %%
# ─── Callbacks ───────────────────────────────────────────────────────────────
@callback(
    Output("vintage-chart", "figure"),
    Input("due-select", "value"),
    Input("window-slider", "value"),
)
def update_vintage_chart(due, min_window):
    """只负责画图。

    以前这个图、表、表头挤在同一个回调的三个 Output 上，切一次 Numerator/Denominator
    也要把 61 条曲线重建一遍并重新序列化（响应 116–123KB）。拆开之后切 tab 只发表格数据。
    """
    vintage_table, _, _ = filter_vintage_table(due, min_window)
    open_months = sorted(vintage_table.index.tolist(), reverse=True)  # 0 -> -60
    if not open_months:
        return _empty_figure("没有可展示的 cohort")

    x_vals = [int(c) for c in vintage_table.columns]

    fig = go.Figure()
    for i, om in enumerate(open_months):
        row = vintage_table.loc[om]
        if row.dropna().empty:
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        # NaN 不可观测格子 -> 塞 None，plotly 会在该处真正断线，而不是连成一条直线
        y_vals = [float(v) * 100 if pd.notna(v) else None for v in row]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                name=f"open_{om}",
                mode="lines+markers",
                # linear：vintage 是累计比率，spline 会把台阶抹成平滑波动，
                # 掩盖「尾部样本骤减」造成的真实拐点
                marker=dict(size=5, color=color),
                line=dict(width=2, color=color, shape="linear"),
                showlegend=True,
                hovertemplate=(
                    f"<b>open_month={om}</b><br>"
                    "Month on Book: %{x}<br>"
                    "Overdue Rate: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    _base_layout(
        fig,
        x_title="Month on Book",
        y_title="Overdue Rate (%)",
        xaxis={"zeroline": False, "title_standoff": 8},
        yaxis={"tickformat": ".1f", "ticksuffix": "%", "rangemode": "tozero"},
        # 61 条线全部上图例，横排铺在 x 轴下方。b=176 是留给图例的高度：
        # 按当前宽度约 6 行，留够余量后 x 轴标题不会被图例盖住。
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            x=0,
            itemsizing="constant",
            font=dict(size=10, color=MUTED),
            tracegroupgap=2,
        ),
        margin=dict(l=52, r=24, t=40, b=176),
    )
    return fig


@callback(
    Output("vintage-table", "data"),
    Output("vintage-table", "columns"),
    Input("due-select", "value"),
    Input("window-slider", "value"),
    Input("vintage-table-tabs", "value"),
    Input("table-compact", "value"),
)
def update_vintage_table(due, min_window, tab, compact):
    vintage_table, numerator, denominator = filter_vintage_table(due, min_window)

    if tab == "rate":
        source, fmt = vintage_table, lambda v: f"{float(v) * 100:.2f}%"
    elif tab == "numerator":
        source, fmt = numerator, lambda v: f"{int(v):,}"
    else:  # denominator
        source, fmt = denominator, lambda v: f"{int(v):,}"

    table_df = source.reset_index().dropna(how="all", axis=1)

    # 紧凑视图：右侧那些「只有一两个老 cohort 有值」的月份列会让表格横向滚很远，
    # 默认仍然全展示，勾选后才裁剪。
    if compact and "on" in compact:
        keep = [c for c in table_df.columns[1:] if table_df[c].notna().mean() >= 0.5]
        table_df = table_df[["open_month"] + keep]

    for col in table_df.columns[1:]:
        table_df[col] = [fmt(v) if pd.notna(v) else "-" for v in table_df[col]]

    columns = [{"name": str(c), "id": str(c)} for c in table_df.columns]
    table_df.columns = [str(c) for c in table_df.columns]
    return table_df.to_dict("records"), columns


@callback(
    Output("rollrate-chart", "figure"),
    Input("mob-select", "value"),
)
def update_rollrate(mob):
    df_counts, df_rates, col_statuses, col_totals = back_roll_rate(mob)

    all_rows = list(df_rates["status"]) + ["worse_rate", "recover_rate"]

    z, text = [], []
    for row_label in all_rows:
        z_row, text_row = [], []
        for ls in col_statuses:
            totals = col_totals[ls]
            if row_label == "worse_rate":
                worse_statuses = [s for s in col_statuses if s > ls]
                worse_sum = sum(
                    df_counts.loc[df_counts["status"] == s, ls].values[0]
                    for s in worse_statuses
                    if s in df_counts["status"].values and ls in df_counts.columns
                )
                rate = worse_sum / totals if totals > 0 else 0
                z_row.append(float(rate))
                text_row.append(f"{rate:.2%}")

            elif row_label == "recover_rate":
                same_val = (
                    df_counts.loc[df_counts["status"] == ls, ls].values[0]
                    if ls in df_counts["status"].values and ls in df_counts.columns
                    else 0
                )
                same_rate = same_val / totals if totals > 0 else 0
                worse_statuses = [s for s in col_statuses if s > ls]
                worse_sum = sum(
                    df_counts.loc[df_counts["status"] == s, ls].values[0]
                    for s in worse_statuses
                    if s in df_counts["status"].values and ls in df_counts.columns
                )
                worse_rate_val = worse_sum / totals if totals > 0 else 0
                rate = max(0.0, 1 - worse_rate_val - same_rate)
                z_row.append(float(rate))
                text_row.append(f"{rate:.2%}")

            else:
                row_mask = df_rates["status"] == row_label
                if row_mask.any() and ls in df_rates.columns:
                    val = df_rates.loc[row_mask, ls].values[0]
                    z_row.append(float(val) if pd.notna(val) else 0.0)
                    text_row.append(f"{val:.2%}" if pd.notna(val) else "0.00%")
                else:
                    z_row.append(0.0)
                    text_row.append("0.00%")

        z.append(z_row)
        text.append(text_row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_statuses,
            y=all_rows,
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0, "#e8f0fb"],
                [0.25, "#a8c8f0"],
                [0.5, "#3987e5"],
                [0.75, "#1c5cab"],
                [1, "#104281"],
            ],
            showscale=True,
            colorbar=dict(title="Rate", tickformat=".1%"),
            hovertemplate="Current: %{y}<br>Previous: %{x}<br>Rate: %{z:.2%}<extra></extra>",
        )
    )

    _base_layout(
        fig,
        x_title="last_status (previous month)",
        y_title="status (current month)",
        yaxis={"autorange": "reversed"},
        margin=dict(l=80, r=24, t=24, b=48),
    )
    return fig


@callback(Output("observe-chart", "figure"), Input("due-select", "id"))
def update_observe_window(_due_id):
    """CDF 只跟账户的观测长度有关，与 due / min_window 都无关。

    注意这里必须留一个 Input：Dash 2.x 不允许「只有 Output、没有 Input」的回调，
    前端渲染器会在建回调图时抛 "A callback is missing Inputs"，然后**整页所有图表
    都不再更新**（不只是这一张）。所以用 due-select 的 id 当占位输入——id 永不变化，
    等价于「只在首次加载时算一次」，不会因为拖滑块而白重算。
    """
    x, y = observe_window_cdf()

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=2, color="#2a78d6"),
            fill="tozeroy",
            fillcolor="rgba(42,120,214,0.12)",
            hovertemplate="Window: %{x}<br>CDF: %{y:.2%}<extra></extra>",
        )
    )
    _base_layout(
        fig,
        x_title="Window (months)",
        y_title="Cumulative Proportion",
        yaxis={"tickformat": ".0%", "rangemode": "tozero"},
    )
    return fig


@callback(
    Output("survival-chart", "figure"),
    Output("survival-desc", "children"),
    Input("survival-due-tabs", "value"),
    Input("window-slider", "value"),
)
def update_survival(due, min_window):
    """KM 曲线的逾期定义维度由卡片内的选项卡独立控制，不受顶部 due-select 影响。"""
    timeline, surv_probs = survival_curve(due, min_window)
    color = DUE_COLORS.get(due, "#eb6834")
    label = dict(DUE_OPTIONS).get(due, due)
    desc = DUE_DESC.get(due, "")

    if not timeline:
        return _empty_figure(f"{label} · min_window={min_window} 筛选后样本为空"), desc

    fig = go.Figure(
        go.Scatter(
            x=timeline,
            y=surv_probs,
            name=label,
            mode="lines",
            line=dict(width=2, color=color),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.12),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Month on Book: %{x}<br>"
                "Survival: %{y:.2%}<extra></extra>"
            ),
        )
    )
    # 末端存活率标注，切换维度时可直接对比
    fig.add_annotation(
        x=timeline[-1],
        y=surv_probs[-1],
        text=f"{surv_probs[-1]:.2%} @ mob {timeline[-1]}",
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=BORDER,
        ax=-8,
        ay=-24,
        font=dict(size=11, color=MUTED),
        bgcolor="rgba(252,252,251,0.85)",
        borderpad=3,
    )
    _base_layout(
        fig,
        x_title="Month on Book",
        y_title="Survival Probability",
        yaxis={"tickformat": ".0%", "rangemode": "tozero"},
    )
    return fig, desc


# %%
if __name__ == "__main__":
    # 不用 debug=True 的默认 reloader：每次热重启都会重新加载 104 万行（0.7s / 579MB）。
    app.run(
        debug=os.environ.get("DEBUG", "0") == "1",
        port=int(os.environ.get("PORT", 8050)),
        use_reloader=False,
    )
