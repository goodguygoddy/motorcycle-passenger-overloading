#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ─────────────────────────────────────────
GT_CSV       = "annotations/labels.csv"
PRED_CSV     = "annotations/predictions.csv"
ZERO_DIVIDE  = 0  # what to put when zero division happens
OUTPUT_FIG   = None  # e.g. "cmaps.png" to save all plots

# ── Load ground truth (one row per image) ─────────────────
gt = (
    pd.read_csv(GT_CSV)
      .set_index("file")
      .astype(int)
)
labels = gt.columns.tolist()

# ── Load raw predictions (one row per person) ────────────
pred = pd.read_csv(PRED_CSV)

# ── Aggregate to image level: if any person has a flag → image flagged
agg_pred = (
    pred
      .groupby("file")[labels]
      .max()
      .astype(int)
)

# ── Align indices and drop any mismatches ────────────────
common = gt.index.intersection(agg_pred.index)
gt      = gt.loc[common]
agg_pred= agg_pred.loc[common]

print(f"🧪 Evaluating on {len(gt)} images")
print("Labels:", labels)
print()

# ── Classification report ───────────────────────────────
print("===== MULTI-LABEL CLASSIFICATION REPORT =====")
print(
    classification_report(
        gt.values,
        agg_pred.values,
        target_names=labels,
        zero_division=ZERO_DIVIDE
    )
)

# ── Per-label confusion matrices ─────────────────────────
for lbl in labels:
    y_true = gt[lbl]
    y_pred = agg_pred[lbl]
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    plt.figure(figsize=(4,3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No","Yes"],
        yticklabels=["No","Yes"]
    )
    plt.title(f"Confusion Matrix: {lbl}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if OUTPUT_FIG:
        plt.savefig(f"{lbl}_{OUTPUT_FIG}")
    else:
        plt.show()
