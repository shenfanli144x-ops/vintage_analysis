# %%
from pathlib import Path

import numpy as np
import pandas as pd

# 数据文件与模块同目录，不依赖启动时的工作目录：
# 之前用相对路径 "credit_info.csv"，换个目录启动就会重新下载 104.8 万行。
BASE_DIR = Path(__file__).resolve().parent

# %%
# ─── Load & prep data ────────────────────────────────────────────────────────
# 本模块只负责「算一次」：把 credit_info 和派生表 tmp 建好。
# 约定：tmp / credit_info 是只读基线，下游任何函数都不得原地修改，
# 需要加工时先 copy（或取所需列的子集 copy）再算。
LOCAL_PATH = BASE_DIR / "credit_info.csv"
print("[data_loader] starting")

# 只读需要的三列：
#   1) 磁盘上的 credit_info.csv 曾被 credit_analysis.py 的 to_csv 反复写脏，
#      多了 Unnamed: 0.1 / Unnamed: 0 两个索引列，usecols 从读取阶段就挡掉，
#      tmp 少两列 int64（约 -16MB），后面也不会有人误用。
#   2) STATUS 只有 10 个取值，object 类型在 104 万行上是 100 万个 Python 字符串
#      （约 50MB），转 category 后只存一份字典 + 1 字节码（约 -45MB）。
_USECOLS = ["ID", "MONTHS_BALANCE", "STATUS"]

if LOCAL_PATH.exists():
    credit_info = pd.read_csv(LOCAL_PATH, usecols=_USECOLS)
    print(f"[data_loader] loaded from local, shape={credit_info.shape}")
else:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    credit_info = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rikdifos/credit-card-approval-prediction",
        "credit_record.csv",
    )
    credit_info = credit_info[_USECOLS]
    credit_info.to_csv(LOCAL_PATH, index=False)
    print(f"[data_loader] downloaded and saved, shape={credit_info.shape}")

credit_info["STATUS"] = credit_info["STATUS"].astype("category")

# 每个 ID 的第一次记录 = open_month，最后一次记录 = end_month；
# month_on_book 是相对 open_month 的账龄，window 是该客户被观察到的总账龄长度。
# due00 / due30 / due60 是三个逾期口径的原始（逐月）标记，累计口径由下游 copy 后自行计算。
tmp = credit_info.assign(
    open_month=lambda x: x.groupby("ID").MONTHS_BALANCE.transform("min"),
    end_month=lambda x: x.groupby("ID").MONTHS_BALANCE.transform("max"),
).assign(
    month_on_book=lambda x: x.MONTHS_BALANCE - x.open_month,
    window=lambda x: x.end_month - x.open_month,
    due00=lambda x: np.where(x.STATUS.isin(["X", "C"]), 0, 1),
    due30=lambda x: np.where(x.STATUS.isin(["X", "C", "0"]), 0, 1),
    due60=lambda x: np.where(x.STATUS.isin(["X", "C", "0", "1"]), 0, 1),
)

MAX_WINDOW = int(tmp["window"].max())
ALL_MOBS = sorted(tmp["month_on_book"].unique().tolist())
print(
    f"[data_loader] tmp={tmp.shape}, MAX_WINDOW={MAX_WINDOW}, "
    f"ALL_MOBS count={len(ALL_MOBS)}"
)
