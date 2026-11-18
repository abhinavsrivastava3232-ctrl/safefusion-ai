"""
Offline evaluation of SafeFusion decision logs against clip-level ground truth.

Usage:
  python evaluate.py
  python evaluate.py --log logs/decisions.csv --truth ground_truth.csv --out eval_results
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["CRITICAL", "HIGH", "WARNING", "SAFE"]
SEVERITY = {"CRITICAL": 3, "HIGH": 2, "WARNING": 1, "SAFE": 0}
MIN_CONSECUTIVE = 5
WARMUP_SECONDS = 2.0
SESSION_GAP_SECONDS = 5.0
VISION_WEAPON_TAU = 0.4
REQUIRED_LOG_COLUMNS = [
    "timestamp",
    "clip_name",
    "c_weapon_smoothed",
    "c_weapon_raw",
    "c_person_smoothed",
    "p_distress",
    "visual_empty",
    "camera_covered",
    "manual_override",
    "threat_state",
    "visual_ms",
    "audio_ms",
    "fusion_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SafeFusion decision logs")
    parser.add_argument("--log", default="logs/decisions.csv")
    parser.add_argument("--truth", default="ground_truth.csv")
    parser.add_argument("--out", default="eval_results")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str, kind: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        print(f"Missing {kind} file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {kind} file {path}: {exc}")
        return None


def missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def clip_level_from_states(states: list[str]) -> str:
    """Most severe state that appears for at least MIN_CONSECUTIVE consecutive rows."""
    best = "SAFE"
    best_sev = SEVERITY[best]
    i = 0
    n = len(states)
    while i < n:
        j = i + 1
        while j < n and states[j] == states[i]:
            j += 1
        if (j - i) >= MIN_CONSECUTIVE:
            sev = SEVERITY.get(str(states[i]), 0)
            if sev > best_sev:
                best_sev = sev
                best = str(states[i])
        i = j
    return best


def vision_only_row_state(c_weapon_smoothed: float) -> str:
    try:
        value = float(c_weapon_smoothed)
    except (TypeError, ValueError):
        return "SAFE"
    return "CRITICAL" if value > VISION_WEAPON_TAU else "SAFE"


def preprocess(log_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = missing_columns(log_df, REQUIRED_LOG_COLUMNS)
    if missing:
        print(f"Warning: log is missing columns: {missing}. Continuing with what is available.")

    if "manual_override" in log_df.columns:
        before = len(log_df)
        log_df = log_df[log_df["manual_override"].astype(str).str.lower() == "none"].copy()
        print(f"Dropped {before - len(log_df)} rows with a manual override.")
    else:
        print("Warning: column 'manual_override' is missing; cannot drop override rows.")

    if "mode" in log_df.columns:
        before = len(log_df)
        log_df = log_df[log_df["mode"].astype(str) != "vision_only"].copy()
        print(f"Dropped {before - len(log_df)} rows with mode == vision_only.")

    if "timestamp" not in log_df.columns:
        print("Warning: column 'timestamp' is missing; cannot apply warm-up or session checks.")
    else:
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
        n_bad = int(log_df["timestamp"].isna().sum())
        if n_bad:
            print(f"Warning: dropped {n_bad} rows with unparseable timestamps.")
            log_df = log_df.dropna(subset=["timestamp"])
        if "clip_name" in log_df.columns and not log_df.empty:
            kept = []
            for clip, group in log_df.groupby("clip_name", dropna=False):
                group = group.sort_values("timestamp")
                t0 = group["timestamp"].iloc[0]
                kept.append(group[group["timestamp"] >= t0 + pd.Timedelta(seconds=WARMUP_SECONDS)])
            log_df = pd.concat(kept, ignore_index=True) if kept else log_df.iloc[0:0]
            print(f"After {WARMUP_SECONDS:.0f}s warm-up drop: {len(log_df)} rows remain.")

            for clip, group in log_df.groupby("clip_name", dropna=False):
                times = group.sort_values("timestamp")["timestamp"]
                if len(times) < 2:
                    continue
                gaps = times.diff().dt.total_seconds().dropna()
                if (gaps > SESSION_GAP_SECONDS).any():
                    print(
                        f"Warning: clip_name '{clip}' has a gap > {SESSION_GAP_SECONDS:.0f}s "
                        "between consecutive rows (possible mixed sessions)."
                    )

    if "clip_name" not in truth_df.columns or "expected_state" not in truth_df.columns:
        print("Warning: ground_truth.csv needs clip_name and expected_state columns.")
        return log_df, truth_df

    truth_df = truth_df.copy()
    truth_df["clip_name"] = truth_df["clip_name"].astype(str)
    truth_df["expected_state"] = truth_df["expected_state"].astype(str).str.upper()

    if "clip_name" not in log_df.columns:
        print("Warning: column 'clip_name' is missing from the log.")
        return log_df, truth_df

    log_df = log_df.copy()
    log_df["clip_name"] = log_df["clip_name"].astype(str)
    truth_names = set(truth_df["clip_name"])
    log_names = set(log_df["clip_name"])
    missing_data = sorted(truth_names - log_names)
    extra_logged = sorted(log_names - truth_names)
    if missing_data:
        print(f"Clips in ground_truth.csv with no log data: {missing_data}")
    else:
        print("All ground-truth clips have at least some log rows.")
    if extra_logged:
        print(f"Logged clips not in ground_truth.csv (ignored for scoring): {extra_logged}")

    log_df = log_df[log_df["clip_name"].isin(truth_names)].copy()
    return log_df, truth_df


def predictions_for_clips(log_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if log_df.empty or "clip_name" not in log_df.columns:
        return pd.DataFrame(rows)

    sort_cols = ["clip_name"]
    if "timestamp" in log_df.columns:
        sort_cols.append("timestamp")
    ordered = log_df.sort_values(sort_cols)

    truth_map = dict(zip(truth_df["clip_name"], truth_df["expected_state"]))
    for clip, group in ordered.groupby("clip_name", sort=False):
        expected = truth_map.get(clip, "")
        if "threat_state" in group.columns:
            sf_states = group["threat_state"].astype(str).str.upper().tolist()
            sf_pred = clip_level_from_states(sf_states)
        else:
            print(f"Warning: clip '{clip}' has no threat_state column; scoring SAFE.")
            sf_pred = "SAFE"

        if "c_weapon_smoothed" in group.columns:
            vo_states = [vision_only_row_state(v) for v in group["c_weapon_smoothed"].tolist()]
            vo_pred = clip_level_from_states(vo_states)
        else:
            print(f"Warning: clip '{clip}' has no c_weapon_smoothed; vision-only = SAFE.")
            vo_pred = "SAFE"

        rows.append(
            {
                "clip_name": clip,
                "expected_state": expected,
                "safefusion_pred": sf_pred,
                "vision_only_pred": vo_pred,
            }
        )
    return pd.DataFrame(rows)


def class_metrics(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def save_confusion(y_true: list[str], y_pred: list[str], out_prefix: str, title: str) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(out_prefix + ".csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title(title)
    vmax = max(int(matrix.max()), 1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if matrix[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=140)
    plt.close(fig)
    return matrix


def time_to_first_alert(log_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if log_df.empty or "timestamp" not in log_df.columns or "threat_state" not in log_df.columns:
        print("Warning: cannot compute time to first alert (need timestamp and threat_state).")
        return pd.DataFrame(columns=["clip_name", "expected_state", "ttfa_seconds"])

    truth_map = dict(zip(truth_df["clip_name"], truth_df["expected_state"]))
    for clip, group in log_df.sort_values("timestamp").groupby("clip_name"):
        expected = str(truth_map.get(clip, "SAFE")).upper()
        if expected == "SAFE":
            continue
        t0 = group["timestamp"].iloc[0]
        alert = group[group["threat_state"].astype(str).str.upper() != "SAFE"]
        if alert.empty:
            ttfa = np.nan
        else:
            ttfa = (alert["timestamp"].iloc[0] - t0).total_seconds()
        rows.append({"clip_name": clip, "expected_state": expected, "ttfa_seconds": ttfa})
    return pd.DataFrame(rows)


def audio_threshold_analysis(log_df: pd.DataFrame, truth_df: pd.DataFrame, out_dir: str) -> Optional[float]:
    if log_df.empty or "p_distress" not in log_df.columns:
        print("Warning: cannot run audio threshold analysis (missing p_distress).")
        return None

    truth_map = dict(zip(truth_df["clip_name"], truth_df["expected_state"]))
    recs = []
    for clip, group in log_df.groupby("clip_name"):
        expected = str(truth_map.get(clip, "")).upper()
        pmax = pd.to_numeric(group["p_distress"], errors="coerce").max()
        recs.append({"clip_name": clip, "expected_state": expected, "max_p_distress": pmax})
    peaks = pd.DataFrame(recs).dropna(subset=["max_p_distress"])
    if peaks.empty:
        print("Warning: no p_distress values for audio analysis.")
        return None

    distress = peaks[peaks["expected_state"].isin(["HIGH", "WARNING"])]
    safe = peaks[peaks["expected_state"] == "SAFE"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not distress.empty:
        ax.hist(distress["max_p_distress"], bins=10, range=(0, 1), alpha=0.6, label="HIGH/WARNING", color="#d35400")
    if not safe.empty:
        ax.hist(safe["max_p_distress"], bins=10, range=(0, 1), alpha=0.6, label="SAFE", color="#2980b9")
    ax.set_xlabel("max p_distress (after warm-up)")
    ax.set_ylabel("clips")
    ax.set_title("Audio peak by expected class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "audio_max_p_distress_hist.png"), dpi=140)
    plt.close(fig)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep_rows = []
    for thr in thresholds:
        if len(distress):
            recall = float((distress["max_p_distress"] > thr).mean())
        else:
            recall = np.nan
            print("Warning: no HIGH/WARNING clips for audio recall.")
        if len(safe):
            fpr = float((safe["max_p_distress"] > thr).mean())
        else:
            fpr = np.nan
            print("Warning: no SAFE clips for audio false-positive rate.")
        balance = (recall - fpr) if pd.notna(recall) and pd.notna(fpr) else np.nan
        sweep_rows.append(
            {
                "threshold": thr,
                "recall_distress": recall,
                "false_positive_rate_safe": fpr,
                "recall_minus_fpr": balance,
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(out_dir, "audio_threshold_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["threshold"], sweep["recall_distress"], marker="o", label="Recall (HIGH/WARNING)")
    ax.plot(sweep["threshold"], sweep["false_positive_rate_safe"], marker="o", label="FPR (SAFE)")
    ax.set_xlabel("p_distress threshold")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Audio threshold sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "audio_threshold_sweep.png"), dpi=140)
    plt.close(fig)

    valid = sweep.dropna(subset=["recall_minus_fpr"])
    if valid.empty:
        print("Warning: could not pick an audio threshold (need both distress and SAFE clips).")
        return None
    best_row = valid.loc[valid["recall_minus_fpr"].idxmax()]
    best = float(best_row["threshold"])
    print(
        f"Suggested audio threshold: {best:.2f} "
        f"(recall={best_row['recall_distress']:.3f}, FPR={best_row['false_positive_rate_safe']:.3f}, "
        f"recall-FPR={best_row['recall_minus_fpr']:.3f})"
    )
    return best


def latency_analysis(log_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    cols = ["visual_ms", "audio_ms", "fusion_ms"]
    present = [c for c in cols if c in log_df.columns]
    if not present:
        print("Warning: no latency columns found.")
        return pd.DataFrame()

    numeric = log_df.copy()
    for c in present:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")

    if "visual_ms" in numeric.columns:
        vis = numeric["visual_ms"].dropna()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if vis.empty:
            print("Warning: visual_ms has no numeric values for the histogram.")
        else:
            ax.hist(vis, bins=20, color="#34495e", alpha=0.85)
        ax.set_xlabel("visual_ms")
        ax.set_ylabel("rows")
        ax.set_title("Visual inference latency (after warm-up)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "latency_visual_ms_hist.png"), dpi=140)
        plt.close(fig)

    stats = []
    for c in cols:
        if c not in numeric.columns:
            print(f"Warning: latency column '{c}' is missing.")
            stats.append({"metric": c, "mean": np.nan, "std": np.nan, "n": 0})
            continue
        series = numeric[c].dropna()
        stats.append(
            {
                "metric": c,
                "mean": float(series.mean()) if len(series) else np.nan,
                "std": float(series.std(ddof=1)) if len(series) > 1 else (0.0 if len(series) == 1 else np.nan),
                "n": int(len(series)),
            }
        )
    table = pd.DataFrame(stats)
    table.to_csv(os.path.join(out_dir, "latency_stats.csv"), index=False)
    return table


def write_summary(
    out_dir: str,
    pred_df: pd.DataFrame,
    sf_acc: Optional[float],
    vo_acc: Optional[float],
    ttfa_df: pd.DataFrame,
    best_thr: Optional[float],
    latency_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("SafeFusion evaluation summary")
    lines.append("")
    if pred_df.empty:
        lines.append("Clips per expected class: (none)")
    else:
        counts = pred_df["expected_state"].value_counts().reindex(LABELS, fill_value=0)
        lines.append("Clips per expected class:")
        for label in LABELS:
            lines.append(f"  {label}: {int(counts[label])}")
    lines.append("")
    lines.append(f"SafeFusion accuracy: {sf_acc if sf_acc is None else f'{sf_acc:.4f}'}")
    lines.append(f"Vision-only accuracy: {vo_acc if vo_acc is None else f'{vo_acc:.4f}'}")

    if ttfa_df.empty or ttfa_df["ttfa_seconds"].dropna().empty:
        lines.append("Mean time to first alert: n/a")
    else:
        mean_ttfa = float(ttfa_df["ttfa_seconds"].mean())
        lines.append(f"Mean time to first alert: {mean_ttfa:.3f} s")

    if best_thr is None:
        lines.append("Suggested audio threshold: n/a")
    else:
        lines.append(f"Suggested audio threshold: {best_thr:.2f}")

    lines.append("Mean latencies:")
    if latency_df.empty:
        lines.append("  n/a")
    else:
        for _, row in latency_df.iterrows():
            mean = row["mean"]
            mean_s = "n/a" if pd.isna(mean) else f"{mean:.3f} ms"
            lines.append(f"  {row['metric']}: {mean_s}")

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def evaluate(log_path: str, truth_path: str, out_dir: str) -> int:
    ensure_dir(out_dir)
    log_df = load_csv(log_path, "log")
    truth_df = load_csv(truth_path, "ground truth")
    if log_df is None or truth_df is None:
        print("Cannot evaluate without both a log file and ground_truth.csv.")
        write_summary(out_dir, pd.DataFrame(), None, None, pd.DataFrame(), None, pd.DataFrame())
        return 1

    log_df, truth_df = preprocess(log_df, truth_df)
    pred_df = predictions_for_clips(log_df, truth_df)
    if pred_df.empty:
        print("No clips left to score.")
        latency_df = latency_analysis(log_df, out_dir)
        write_summary(out_dir, pred_df, None, None, pd.DataFrame(), None, latency_df)
        return 0

    pred_df.to_csv(os.path.join(out_dir, "clip_predictions.csv"), index=False)

    y_true = pred_df["expected_state"].tolist()
    y_sf = pred_df["safefusion_pred"].tolist()
    y_vo = pred_df["vision_only_pred"].tolist()

    sf_acc = float(accuracy_score(y_true, y_sf))
    vo_acc = float(accuracy_score(y_true, y_vo))
    print(f"SafeFusion accuracy: {sf_acc:.4f}")
    print(f"Vision-only accuracy: {vo_acc:.4f}")

    sf_cls = class_metrics(y_true, y_sf)
    vo_cls = class_metrics(y_true, y_vo)
    sf_cls.to_csv(os.path.join(out_dir, "safefusion_class_metrics.csv"), index=False)
    vo_cls.to_csv(os.path.join(out_dir, "vision_only_class_metrics.csv"), index=False)

    save_confusion(y_true, y_sf, os.path.join(out_dir, "safefusion_confusion"), "SafeFusion confusion matrix")
    save_confusion(y_true, y_vo, os.path.join(out_dir, "vision_only_confusion"), "Vision-only confusion matrix")

    comparison = pd.DataFrame(
        {
            "metric": ["accuracy"] + [f"recall_{c}" for c in LABELS],
            "safefusion": [sf_acc] + sf_cls["recall"].tolist(),
            "vision_only": [vo_acc] + vo_cls["recall"].tolist(),
        }
    )
    comparison.to_csv(os.path.join(out_dir, "comparison_safefusion_vs_vision.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(comparison))
    width = 0.35
    ax.bar(x - width / 2, comparison["safefusion"], width, label="SafeFusion", color="#1f6feb")
    ax.bar(x + width / 2, comparison["vision_only"], width, label="Vision-only", color="#8b949e")
    ax.set_xticks(x, comparison["metric"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("SafeFusion vs vision-only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_safefusion_vs_vision.png"), dpi=140)
    plt.close(fig)

    ttfa_df = time_to_first_alert(log_df, truth_df)
    if not ttfa_df.empty:
        ttfa_df.to_csv(os.path.join(out_dir, "time_to_first_alert.csv"), index=False)
        summary = (
            ttfa_df.groupby("expected_state")["ttfa_seconds"]
            .agg(["mean", "std", "count"])
            .reindex(LABELS)
            .reset_index()
        )
        summary.to_csv(os.path.join(out_dir, "time_to_first_alert_by_class.csv"), index=False)
        print("Time to first alert (seconds) by expected_state:")
        print(summary.to_string(index=False))
    else:
        print("No non-SAFE clips for time-to-first-alert.")

    best_thr = audio_threshold_analysis(log_df, truth_df, out_dir)
    latency_df = latency_analysis(log_df, out_dir)
    write_summary(out_dir, pred_df, sf_acc, vo_acc, ttfa_df, best_thr, latency_df)
    return 0


def main() -> int:
    args = parse_args()
    return evaluate(args.log, args.truth, args.out)


if __name__ == "__main__":
    sys.exit(main())
