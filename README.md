# 信用卡账户逾期风险分析看板

**Vintage Analysis + Backward Roll Rate + Survival Model**

基于 Kaggle *Credit Card Approval Prediction* 数据构建的账户级风险演化分析系统。以账龄（Month on Book, MOB）为时间轴，用四种互补的统计视角刻画逾期风险如何随账户老化而累积、迁徙与衰减，并通过 Dash 交互看板交付给风控策略人员使用。

---

## 一、项目定位

风险策略里最常被问到三个问题，本项目分别用不同方法回答：

| 业务问题 | 对应方法 | 输出 |
| --- | --- | --- |
| 不同开户批次的账户，风险水平是否在恶化？ | **Vintage 分析** | 61 条 cohort 累计逾期率曲线 + 分子/分母明细表 |
| 逾期状态如何逐月迁徙？有多少能自愈？ | **回溯 Roll Rate** | 状态迁移矩阵（含坏账迁徙率、回收率） |
| 什么账龄开始进入高危期？长期表现如何？ | **Kaplan-Meier 生存分析** | 未逾期存活概率曲线 |
| 数据能支撑到多少账龄？晚期点位的结论可信吗？ | **观测窗口 CDF** | 账户观测长度分布，用于判断样本充足度 |

**技术栈**：Python · pandas · Plotly · Dash · lifelines · pytest

---

## 二、数据集

| 文件 | 内容 | 规模 |
| --- | --- | --- |
| `credit_info.csv` | 账户逐月还款状态（`ID`, `MONTHS_BALANCE`, `STATUS`） | 104.8 万行 / 45,985 个账户 / 最长 61 期 |

本项目只使用账户账本数据（B 卡行为维度），不涉及申请信息（A 卡）字段。

`data_loader.py` 优先读取本地 `credit_info.csv`；文件缺失时通过 `kagglehub` 自动下载并落盘，保证开箱即跑。

**STATUS 语义**

| STATUS | 含义 |
| --- | --- |
| `0` | 1–29 天逾期 |
| `1` – `5` | 逾期 30 / 60 / 90 / 120 / 150+ 天 |
| `C` | 已结清（closed） |
| `X` | 当月无账（no record） |

**三套逾期口径**（逐月原始标记，累计由下游按需构造）

| 字段 | 事件定义 | 业务含义 |
| --- | --- | --- |
| `due00` | `STATUS ∉ {X, C}` | 任何逾期（any delinquency） |
| `due30` | `STATUS ∉ {X, C, 0}` | 逾期 30 天及以上（M1+） |
| `due60` | `STATUS ∉ {X, C, 0, 1}` | 逾期 60 天及以上（M2+） |

同一套分析在三种口径下并行呈现，用于区分「轻度拖欠」与「实质性坏账」的驱动差异——两者的结论方向可能完全相反（见第八节）。

---

## 三、派生字段

对每个 `ID` 计算账龄坐标系：

```
open_month      = min(MONTHS_BALANCE)            开户月，取值 0 ~ -60
end_month       = max(MONTHS_BALANCE)            最后观测月
month_on_book   = MONTHS_BALANCE - open_month    账龄，从 0 起算
window          = end_month - open_month         该账户被观测到的总账龄
```

`open_month` 即 cohort 标识（0 = 最新批次，-60 = 最早批次），`window` 决定每个 cohort 在各 MOB 上是否可观测。

---

## 四、方法论

### 4.1 Vintage 分析

按开户月分组，追踪每个 cohort 在各账龄上的**累计**逾期率。

对同一账户的 `due` 序列按 MOB 升序取 `cummax`——**一旦逾期，此后各期均计为逾期**，得到 ever-overdue 标记：

\[
\text{due}^{\text{cum}}_{i,t} = \max_{s \le t} \text{due}_{i,s}
\]

\[
\text{VintageRate}(c, t) = \frac{\sum_{i \in \text{cohort } c} \text{due}^{\text{cum}}_{i,t}}{\#\{i \in \text{cohort } c \text{ 在 } t \text{ 时在账}\}}
\]

- **`min_window` 过滤**：只保留观测长度 ≥ 阈值的账户，避免短账龄账户在早期点位拉低分母
- 输出三份数据：比率表、分子表、分母表（看板可切换查看，便于核对每个点位的样本量）

### 4.2 回溯 Roll Rate

刻画「上月状态 → 本月状态」的迁移分布，可整体计算或按 `month_on_book` 切片：

- 行 = 当期 `STATUS`，列 = 上期 `STATUS`
- 单元格 = 迁移占比，**分母为列合计**（即「上期处于该状态的全部人」），因此每列加总为 1
- 列中剔除 `C`（销户）与 `X`（当月无账），只保留真实逾期状态
- 额外两行汇总：
  - `worse_rate` = 向下迁徙（状态恶化）占比
  - `recover_rate` = \(\max(0,\ 1 - \text{worse} - \text{same})\)，即改善占比

### 4.3 观测窗口 CDF

每个账户只贡献一次，统计 `window` 的经验分布：

\[
\text{CDF}(w) = \frac{\#\{i : \text{window}_i \le w\}}{N}
\]

曲线尾部趋于平缓的点位，意味着更长账龄已几乎没有新增样本——这是判断 Vintage 曲线右端可信度的直接依据。该分布只与观测长度有关，与逾期定义无关。

### 4.4 Kaplan-Meier 生存分析

把「首次逾期」视为事件，估计账户在账龄维度上的存活（未逾期）概率：

- **事件时点** = 该账户首次 `due == 1` 的 `month_on_book`
- **删失时点** = 未逾期账户的最后观测 `month_on_book`
- **事件标记** = 是否曾出现 `due == 1`

由 `lifelines.KaplanMeierFitter` 拟合，可切换三套口径对比。

> **口径隔离**：KM 直接使用**未 cummax 的逐月原始标记**。事件时点取 `min`、删失时点取 `max`、事件标记取 `max`，三个统计量在 cummax 前后完全等价，因此 KM 结果不会被 Vintage 的 ever-overdue 口径污染，也与函数调用顺序无关。

> 本节只讲方法思路。**每个数值的分子 / 分母 / 过滤条件 / 缺失处理**逐项规定见 [附录 A：指标口径定义](#附录-a指标口径定义)。

---

## 五、系统架构

```
credit_info.csv
      │  usecols 只读 3 列 / STATUS → category
      ▼
data_loader.py  ──►  credit_info（原始） + tmp（派生字段，只读基线）
      │
      ▼
analysis.py     ──►  4 个纯函数，lru_cache 缓存（命中后返回 .copy()）
      │                filter_vintage_table / back_roll_rate
      │                observe_window_cdf / survival_curve
      ▼
app.py          ──►  4 个独立回调 ──►  Plotly 图表 / DataTable
      │
      ▼
Dash 单页看板（Vintage → Roll Rate → CDF & KM）
```

**页面结构**

1. **Vintage Analysis** — 逾期口径下拉 + `min_window` 滑块；61 条 cohort 曲线（8 色循环 + 图例）+ 可切换 `%` / 分子 / 分母的明细表（排序、冻结首列、紧凑视图）
2. **Backward Roll Rate** — `month_on_book` 选择（`ALL` 或 5–60）；状态迁移热力图，逐格标注占比
3. **& 4. Observe Window & Survival Analysis** — 左：观测窗口 CDF；右：KM 曲线（卡片内独立切换 `due00/30/60`，与顶部口径解耦）

---

## 六、工程实践

分析层的正确性比图表好看重要得多，以下四条约束写进了代码注释，并由测试守护：

1. **`tmp` 是只读基线**。所有加工先 `copy()` 所需列，绝不原地写回。历史事故：把 `cummax` 结果回写 `tmp`，导致后续所有分析口径被静默改变，且极难复现。
2. **缓存必须返回副本**。`lru_cache` 存的是对象引用，公开入口一律 `.copy()` 后返回，否则某个回调原地改一列就会永久污染缓存；`min_window` 统一 `int()` 规范化，避免 `24` 与 `24.0` 占两份缓存。
3. **不可观测格子必须是 `NaN`，绝不填 0**。某 cohort 还没走到该 MOB 时分母缺失，属不可观测区域：图上断线、表中显示 `-`。曾用 `replace({np.nan: 0})` 填零，导致 49.2% 的格子伪造出「逾期率归零」的假象。
4. **`STATUS` 是 `category`，`groupby` 必须显式 `observed=True`**。否则默认会把状态类别的笛卡尔积全部展开，矩阵凭空多出无人空行。

**性能优化（本机实测，非估算）**

| 优化项 | 改动 | 效果 |
| --- | --- | --- |
| 分析层结果缓存 | `lru_cache` + 出口 `.copy()` | Vintage 切 tab **365ms → 8ms**；Roll rate **565 → 41ms**；KM **840 → 41ms** |
| 拆分 Vintage 回调 | figure / table 两个 Output 分离 | 切 tab 响应 **116KB → 36KB**，不再重建 61 条曲线 |
| 数据层瘦身 | `usecols` 剔除历史脏列 + `STATUS → category` | `tmp` **146MB → 73MB**，导入峰值同步下降 |
| 启动方式 | `PORT` / `DEBUG` 环境变量 + `use_reloader=False` | 改代码不再重启进程重新加载百万行 |
| 交互体验 | `dcc.Loading` 加载态 / 线性曲线 / 表格排序冻结 | 消除「点了没反应」，台阶不被 spline 抹平 |

优化全程**数值零变化**：Vintage（9 组）、KM（9 组）、Roll rate（4 组）、CDF 与改造前基线逐值一致（`allclose(equal_nan=True)`）。完整方案与踩坑记录见 `app_optimization_plan.md`。

---

## 七、运行与测试

```bash
pip install -r requirements.txt

python3 app.py                      # http://127.0.0.1:8050
PORT=8060 DEBUG=1 python3 app.py    # 自定义端口 / 开启调试
```

```bash
python3 -m pytest test_app.py -q    # 51 个用例，约 7s
```

测试覆盖 4 个回调 × 边界参数（`min_window = 0/24/55/60`、`mob = ALL/5/…/60`、3 口径 × 3 tab），以及这些易复发的约定：不可观测格子必须是 `None`（断线）、CDF 单调且终值为 1、KM 曲线不得上升、三口径曲线必须互不相同、缓存副本隔离、`min_window` 整数归一化、`tmp` 调用后逐值不变、每个回调必须有 Input（Dash 2.x 缺失会导致**整页**回调失效）。

测试与看板共用同一套平铺模块（`import analysis` / `import app`），在项目根目录直接执行即可，无需设置 `PYTHONPATH`。

**目录结构**

```
.
├── data_loader.py           # 数据加载 + 派生表 tmp（只读基线）
├── analysis.py              # 四个分析函数（纯函数 + 缓存）
├── app.py                   # Dash 布局、样式与回调
├── test_app.py              # 回调回归测试
├── app_optimization_plan.md # 性能优化方案与落地记录
├── README.md
├── requirements.txt
├── .gitignore
└── credit_info.csv
```

---

## 八、关键发现

1. **风险随账龄单调累积且 cohort 间分层明显**：老批次（`open_month` 更负）在相同 MOB 上的累计逾期率显著高于新批次，符合账户老化规律；曲线斜率在早期（MOB 0–12）最陡，进入 MOB 24 后趋缓。
2. **口径决定结论方向**：`due30/60` 下，逾期客户往往仍在账上，留存样本比率被抬高；而 `due00`（任何逾期）里，有逾期记录的客户更容易销户离账，「仍在账」样本被清洗成偏好客户，比率偏低。同一现象在两种口径下符号相反，**因此三口径并行展示是分析的前提，不是冗余**。
3. **尾部点位不可轻信**：观测窗口 CDF 显示大量账户在 MOB 30 之后已无样本；在「当期在账人数」做分母时，实测 45/61 行出现累计逾期率下降（最大单步 0.95pp），这是样本流失（attrition bias）而非资产质量改善。
4. **KM 与 Vintage 互为校验**：ever-overdue 口径的 vintage 曲线不应低于 KM 的累计事件率，两者在中期点位差值过大即说明分母口径有问题。

---

## 九、已知局限与后续方向

当前版本**刻意保留**两处原始口径（详见 `app_optimization_plan.md` P0-3 / P0-4），因为改动会改变全部数值，需要业务侧先确认定义：

- **Vintage 分母为「当期在账人数」而非 cohort 全量**，属条件分母。若改为 cohort 全量、分子按首次逾期月累计，可做到 0/61 行下降，但三个维度分别变动 +4.74pp / −2.05pp / −2.27pp，方向不统一。**在当前展示下解读尾部曲线，必须注明分母定义**，否则会把样本 attrition 误读为质量改善。
- **`recover_rate = 1 − worse − same` 是残差定义**，且 `X`（当月无账）本身不是状态转移。四项完整分解（same / worse / better / closed）尚未落地，当前 `status='0'` 列的「回收率」实际等于销户率。

后续可做：

- [ ] Vintage 标准口径切换 + 双口径对照视图
- [ ] Roll Rate 四项分解与口径自检（四项之和恒为 1）
- [ ] 按 `min_window` 自动标注「样本量不足」的不可信点位

---

## 附录 A：指标口径定义

本节逐个规定看板上每个数值的分子、分母、过滤条件、单位与缺失处理。**同一名称的指标在不同模块里口径不同时，以本节为准**；解读任何图表前先确认它落在哪一档。

### A.0 全局约定

| 项 | 规定 |
| --- | --- |
| 数据粒度 | `credit_info` 一行 = 一个账户的一个自然月（`ID` × `MONTHS_BALANCE`） |
| 时间轴 | 一律用 `month_on_book`（账龄，开户月 = 0），**不用**日历月 |
| 状态缺失 | 某月无记录时该行可能缺失（或为 `X`），不做插值、不向前填充 |
| 缺失展示 | 表格显示 `-`，曲线**断线**（`None`），热力图按 0 处理 |
| 数值单位 | 内部一律存 0–1 小数，仅在 UI 层 ×100 加 `%` |
| 过滤时机 | `min_window` 在**账户级**生效：命中阈值的账户保留其全部月份，而非逐行过滤 |

### A.1 基础派生字段（`data_loader.py`）

| 字段 | 计算式 | 粒度 | 边界与说明 |
| --- | --- | --- | --- |
| `open_month` | `min(MONTHS_BALANCE)` by `ID` | 账户 | 即 cohort 标识，取值 0（最新）~ −60（最早） |
| `end_month` | `max(MONTHS_BALANCE)` by `ID` | 账户 | 最后一次观测月 |
| `month_on_book` | `MONTHS_BALANCE − open_month` | 账户-月 | 从 0 起算，最大值 = 该账户 `window` |
| `window` | `end_month − open_month` | 账户 | 观测总长度，0–60；**决定可观测上界** |
| `due00` | `STATUS ∉ {X, C}` → 1 | 账户-月 | 任何逾期；`C`/`X` 记为 0 |
| `due30` | `STATUS ∉ {X, C, 0}` → 1 | 账户-月 | 逾期 30+ 天 |
| `due60` | `STATUS ∉ {X, C, 0, 1}` → 1 | 账户-月 | 逾期 60+ 天 |

> `due*` 在 `tmp` 中**始终是逐月原始标记**，不含累计。累计（`cummax`）只在 Vintage 内部临时构造，不写回 `tmp`。

### A.2 Vintage 累计逾期率

| 项 | 规定 |
| --- | --- |
| 入参 | `due ∈ {due00, due30, due60}`；`min_window ∈ [0, 60]` 整数 |
| 账户过滤 | `window >= min_window`（**大于等于**，含等于） |
| 累计方式 | 账户内按 `month_on_book` 升序取 `cummax(due)`：首次逾期后各期恒为 1 |
| 分子 | 该 `(open_month, month_on_book)` 上累计标记为 1 的账户数（`sum`） |
| 分母 | 该 `(open_month, month_on_book)` 上在账账户数（`ID` 去重计数） |
| 比率 | 分子 ÷ 分母；分母为 0 → `NaN` |
| 不可观测区 | `month_on_book > 该 cohort 的 window 上界` → `NaN`（**不补 0**） |
| 单调性 | **不保证**。分母是当期在账人数（条件分母），账户离账会同时移出分子与分母 |
| 轴顺序 | x = `month_on_book` 0–60；y = cohort，按 `open_month` 降序（0 → −60） |

### A.3 Vintage 明细表三个 tab

| tab | 含义 | 单元格口径 | 显示格式 |
| --- | --- | --- | --- |
| `%` | 累计逾期率 | 同 A.2 比率 | `xx.xx%`，缺失 `-` |
| `Numerator` | 分子 | 同 A.2 分子 | 整数 + 千分位 |
| `Denominator` | 分母 | 同 A.2 分母 | 整数 + 千分位 |
| 紧凑视图 | 按列裁剪 | 仅保留 `非空占比 ≥ 50%` 的月份列 | 默认关闭 |

### A.4 回溯 Roll Rate

| 项 | 规定 |
| --- | --- |
| 分析单元 | 账户的**相邻两个月**：本期 `(ID, t)` 与上期 `(ID, t−1)` 精确配对 |
| 配对方式 | 按 `MONTHS_BALANCE − 1` 内连接；上期 `> open_month`（保证上期已开户） |
| 缺月处理 | 缺月即无法配对，该月对**直接丢弃**，不跨缺口补齐 |
| `mob` 过滤 | `month_on_book == mob`；`ALL` 表示不做账龄过滤（所有月份对汇总） |
| 行 | 本期状态，保留全部取值（含 `C` 销户、`X` 当月无账） |
| 列 | 上期状态，**剔除 `C` 与 `X`**，只保留真实逾期状态 |
| 单元格 | 该 `(本期状态, 上期状态)` 上的**去重账户数** |
| 占比 | 单元格 ÷ **列合计**（即上期处于该状态的全部对象），列内加总 = 1 |
| `worse_rate` | 列内「本期状态比上期更差」的账户数 ÷ 列合计；状态按字符串序比较（`'0'<'1'<…<'5'`，与逾期严重度一致），`C`/`X` 已不在列中 |
| `recover_rate` | `max(0, 1 − worse_rate − same_rate)`，**残差口径**，非独立统计 |
| 空列 | 列合计为 0 时占比记 0，不报错 |

> **`ALL` 口径的分母陷阱**：`ALL` 下同一账户可在同一列的不同月份重复出现，单元格按 `ID` 去重、列合计却是各格去重数之和，因此**列合计不是独立人数**，而是「账户-状态」配对数。按单个 `mob` 查看时，每个账户在该月仅出现一次，列合计才等于「上期处于该状态的人数」。

### A.5 观测窗口 CDF

| 项 | 规定 |
| --- | --- |
| 分析单元 | 账户（`ID` 去重后每人一条） |
| x | `window`（0–60 个月） |
| y | `P(window ≤ x)`，按 `window` 取值的累计占比，终值 = 1.0 |
| 入参 | 无。只与观测长度有关，与 `due` / `min_window` 均无关（回调用 `due-select.id` 占位） |
| 用途 | 判断各 MOB 点位的样本充足度；尾部走平说明更长账龄已无新增样本 |

### A.6 Kaplan-Meier 生存曲线

| 项 | 规定 |
| --- | --- |
| 入参 | `due ∈ {due00, due30, due60}`；`min_window` 来自顶部滑块 |
| 账户过滤 | `window > min_window`（**严格大于**，与 A.2 的 `>=` 差 1 个月） |
| 事件定义 | 该账户在观测期内是否出现过 `due == 1`（`max(due)`） |
| 事件时点 | 首次 `due == 1` 的 `month_on_book`（`min`） |
| 删失时点 | 未发生事件账户的最后观测 `month_on_book`（`max`） |
| 累计标记 | **不使用** `cummax`：直接用逐月原始标记。事件时点取 `min`、删失取 `max`，两者对 `cummax` 不变，故与调用顺序无关 |
| 曲线 | `S(t) = ∏(1 − dᵢ/nᵢ)`，由 `lifelines.KaplanMeierFitter` 拟合；x = 生存函数时间点，y = 未逾期概率 |
| 空样本 | 过滤后为空时返回占位图（不抛异常），例如 `min_window` 取到 60 |

### A.7 UI 展示口径

| 场景 | 规则 |
| --- | --- |
| 百分比 | `值 × 100`，保留 2 位小数（Vintage 表格、Roll Rate 热力图标注） |
| 坐标轴刻度 | Vintage y 轴 `.1f%`；KM / CDF y 轴 `.0%` |
| 计数 | 整数 + 千分位分隔符 |
| 缺失 | 表格 `-`；Vintage 曲线 `None`（真实断线）；Roll Rate 热力图按 0 着色 |
| 图例 | 每条 cohort 一个图例项，命名 `open_{open_month}`，8 色循环 |

### A.8 口径差异备忘（必须知道的三处）

| 位置 | 差异 | 后果 |
| --- | --- | --- |
| Vintage vs KM 的 `min_window` | `>=` 对 `>` | 恰好 `window == min_window` 的账户进入 Vintage 但不进 KM；对比两条曲线时存在 1 个月的口径错位 |
| Vintage 分母 | 当期在账人数 | 45/61 行出现累计率下降（attrition bias），不可读作质量改善 |
| Roll Rate `recover_rate` | 残差 `1 − worse − same` | 等于「改善 + 销户 + 无账」混合体，`status='0'` 列上实际等于销户率 |
