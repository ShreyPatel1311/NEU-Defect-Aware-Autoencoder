"""Seed-aggregated tables from results/runs/*/eval.json.
usage: python scripts/aggregate.py [--groups configs/groups.json] [--readout mid|cls|mean] [--csv out.csv]"""
import argparse, json
from defect_aware_mae.report import table

ap = argparse.ArgumentParser(); ap.add_argument("--groups", default="configs/groups.json")
ap.add_argument("--readout", default=None, help="override each group's own readout"); ap.add_argument("--csv")
a = ap.parse_args()
df = table(json.load(open(a.groups)), readout=a.readout)
print(df.to_markdown(index=False))
if a.csv: df.to_csv(a.csv, index=False)
