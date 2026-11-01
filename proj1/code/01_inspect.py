"""Quick look at the three raw files: fields, label values, and ID overlap."""
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

rt = pd.read_parquet(RAW / "reviewertoo_iclr2025_full.parquet")
print("== ReviewerToo full:", rt.shape)
print(rt.dtypes.to_string())
print("\ndecision values:\n", rt["decision"].value_counts(dropna=False).to_string())
print("\ndecision x has_metareview:\n",
      pd.crosstab(rt["decision"], rt["metareview"].fillna("").str.len() > 0).to_string())
print("\nhas review text:", (rt["reviews"].fillna("").str.len() > 0).sum())
print("citations_serper non-null:", rt["citations_serper"].notna().sum())
print("\nprimary_area (top 25):\n", rt["primary_area"].value_counts().head(25).to_string())

pc = json.load(open(RAW / "papercopilot_iclr2025.json"))
print("\n== Paper Copilot:", type(pc).__name__, len(pc))
first = pc[0] if isinstance(pc, list) else next(iter(pc.values()))
print("fields:", sorted(first.keys()))
pcdf = pd.DataFrame(pc)
for col in ("status", "track", "primary_area"):
    if col in pcdf:
        print(f"\n{col} values:\n", pcdf[col].value_counts(dropna=False).head(30).to_string())

ins = json.load(open(RAW / "iclr2025_insights.json"))
ins_ids = {p["id"] for p in ins}
rt_ids = set(rt["paper_id"])
pc_ids = set(pcdf["id"]) if "id" in pcdf else set()
print("\n== ID overlap")
print("insights:", len(ins_ids), " reviewertoo:", len(rt_ids), " papercopilot:", len(pc_ids))
print("insights & reviewertoo:", len(ins_ids & rt_ids))
print("insights & papercopilot:", len(ins_ids & pc_ids))
print("all three:", len(ins_ids & rt_ids & pc_ids))

if "status" in pcdf and "id" in pcdf:
    m = rt.merge(pcdf[["id", "status"]], left_on="paper_id", right_on="id", how="left")
    print("\nReviewerToo decision x Paper Copilot status:\n",
          pd.crosstab(m["decision"], m["status"].fillna("<missing>")).to_string())
