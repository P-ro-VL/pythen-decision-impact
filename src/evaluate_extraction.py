#!/usr/bin/env python3
"""
Standalone Predicate Extraction Evaluation (Ablation Study)
===========================================================
Evaluates the pure linguistic extraction capability of each LLM by comparing
their extracted predicates against ground truth, independent of the PYTHEN
reasoning pipeline.

Metrics computed:
- Per-predicate: Precision, Recall, F1
- Overall: Macro-F1, Micro-F1, Accuracy
- Per-slice breakdown
- Error details & confusion analysis
"""

import ast
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth" / "hearsay_ground_truth.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "predictions" / "pythen"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "summary_metrics"
DEFAULT_ERROR_DIR = REPO_ROOT / "results" / "error_taxonomy"

PREDICATES = [
    "is_oral_assertion",
    "is_written_assertion",
    "is_non_verbal_assertive_conduct",
    "is_out_of_court",
    "is_introduced_to_prove_truth",
]

PREDICATE_LABELS = {
    "is_oral_assertion": "Oral Assertion",
    "is_written_assertion": "Written Assertion",
    "is_non_verbal_assertive_conduct": "Non-verbal Assertive Conduct",
    "is_out_of_court": "Out of Court",
    "is_introduced_to_prove_truth": "Introduced to Prove Truth",
}

MODEL_FILES = {
    "DeepSeek-V3.2": "pythen_deepseek_v3.2.csv",
    "GPT-OSS-120B": "pythen_gpt_oss_120b.csv",
    "Gemma-4-31B": "pythen_gemma_4_31b.csv",
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load ground truth predicates from CSV."""
    df = pd.read_csv(path)
    for pred in PREDICATES:
        df[pred] = df[pred].astype(str).str.strip().str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        )
    return df


def parse_llm_output(raw_output):
    """
    Parse the llm_output string (Python dict literal or JSON) and extract predicate values.
    Returns a dict of {predicate_name: bool}.
    """
    try:
        parsed = ast.literal_eval(str(raw_output))
        facts = parsed.get("facts", {})
        return {pred: bool(facts.get(pred, False)) for pred in PREDICATES}
    except (ValueError, SyntaxError, TypeError):
        pass
    try:
        fixed = str(raw_output).replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        parsed = json.loads(fixed)
        facts = parsed.get("facts", {})
        return {pred: bool(facts.get(pred, False)) for pred in PREDICATES}
    except Exception:
        return None


def load_model_predictions(model_name: str, model_dir: Path) -> pd.DataFrame:
    """Load a model's CSV and extract predicate predictions from llm_output."""
    filepath = model_dir / MODEL_FILES[model_name]
    df = pd.read_csv(filepath)
    
    predictions = []
    parse_errors = 0
    for _, row in df.iterrows():
        parsed = parse_llm_output(row["llm_output"])
        if parsed is None:
            parse_errors += 1
            parsed = {pred: False for pred in PREDICATES}
        parsed["index"] = int(row["index"])
        parsed["slice"] = row.get("slice", "Unknown")
        predictions.append(parsed)
    
    pred_df = pd.DataFrame(predictions)
    if parse_errors > 0:
        print(f"  ⚠  {model_name}: {parse_errors} rows had parse errors in llm_output")
    return pred_df


# ─── Metrics Computation ─────────────────────────────────────────────────────

def compute_binary_metrics(y_true, y_pred):
    """Compute Precision, Recall, F1, Accuracy for binary arrays."""
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "Support": int(tp + fn),
        "Total": len(y_true),
    }


def evaluate_model(gt_df: pd.DataFrame, pred_df: pd.DataFrame, model_name: str) -> dict:
    """Run full evaluation for a single model against ground truth."""
    merged = pd.merge(gt_df, pred_df, on="index", suffixes=("_gt", "_pred"))
    
    results = {
        "model": model_name,
        "total_samples": len(merged),
        "per_predicate": {},
        "overall": {},
        "per_slice": {},
    }
    
    # 1. Per-predicate metrics
    f1_list = []
    all_y_true = []
    all_y_pred = []
    
    for pred in PREDICATES:
        gt_col = f"{pred}_gt" if f"{pred}_gt" in merged.columns else pred
        pred_col = f"{pred}_pred" if f"{pred}_pred" in merged.columns else pred
        
        y_true = merged[gt_col].values
        y_pred = merged[pred_col].values
        
        metrics = compute_binary_metrics(y_true, y_pred)
        results["per_predicate"][pred] = metrics
        f1_list.append(metrics["F1"])
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    # 2. Overall metrics
    macro_f1 = np.mean(f1_list)
    micro_metrics = compute_binary_metrics(all_y_true, all_y_pred)
    
    # Exact match accuracy (all 5 predicates correct for a sample)
    exact_matches = 0
    for _, row in merged.iterrows():
        match = True
        for pred in PREDICATES:
            gt_val = row[f"{pred}_gt" if f"{pred}_gt" in merged.columns else pred]
            pred_val = row[f"{pred}_pred" if f"{pred}_pred" in merged.columns else pred]
            if gt_val != pred_val:
                match = False
                break
        if match:
            exact_matches += 1
    
    results["overall"] = {
        "Macro_F1": float(macro_f1),
        "Micro_F1": micro_metrics["F1"],
        "Micro_Precision": micro_metrics["Precision"],
        "Micro_Recall": micro_metrics["Recall"],
        "Overall_Accuracy": micro_metrics["Accuracy"],
        "Exact_Match_Ratio": exact_matches / len(merged) if len(merged) > 0 else 0.0,
        "Exact_Matches": exact_matches,
        "Total_Slots": len(all_y_true),
        "Micro_TP": micro_metrics["TP"],
        "Micro_FP": micro_metrics["FP"],
        "Micro_FN": micro_metrics["FN"],
        "Micro_TN": micro_metrics["TN"],
    }
    
    # 3. Per-slice metrics
    slice_col = "slice_gt" if "slice_gt" in merged.columns else "slice"
    if slice_col in merged.columns:
        for slice_name, group in merged.groupby(slice_col):
            slice_f1s = []
            slice_y_true = []
            slice_y_pred = []
            
            for pred in PREDICATES:
                gt_c = f"{pred}_gt" if f"{pred}_gt" in group.columns else pred
                pr_c = f"{pred}_pred" if f"{pred}_pred" in group.columns else pred
                yt = group[gt_c].values
                yp = group[pr_c].values
                m = compute_binary_metrics(yt, yp)
                slice_f1s.append(m["F1"])
                slice_y_true.extend(yt)
                slice_y_pred.extend(yp)
            
            slice_micro = compute_binary_metrics(slice_y_true, slice_y_pred)
            results["per_slice"][str(slice_name)] = {
                "sample_count": len(group),
                "Macro_F1": float(np.mean(slice_f1s)),
                "Micro_F1": slice_micro["F1"],
                "Micro_Precision": slice_micro["Precision"],
                "Micro_Recall": slice_micro["Recall"],
                "Accuracy": slice_micro["Accuracy"],
            }
            
    return results


def generate_error_details(gt_df: pd.DataFrame, pred_df: pd.DataFrame, model_name: str) -> list:
    """Generate detailed list of extraction errors."""
    merged = pd.merge(gt_df, pred_df, on="index", suffixes=("_gt", "_pred"))
    errors = []
    
    for _, row in merged.iterrows():
        idx = row["index"]
        slice_val = row.get("slice_gt", row.get("slice", "Unknown"))
        text_val = row.get("text", "")
        
        for pred in PREDICATES:
            gt_val = row[f"{pred}_gt" if f"{pred}_gt" in merged.columns else pred]
            pred_val = row[f"{pred}_pred" if f"{pred}_pred" in merged.columns else pred]
            
            if gt_val != pred_val:
                error_type = "False Positive" if pred_val and not gt_val else "False Negative"
                errors.append({
                    "sample_index": idx,
                    "slice": slice_val,
                    "predicate": pred,
                    "predicate_label": PREDICATE_LABELS[pred],
                    "ground_truth": bool(gt_val),
                    "predicted": bool(pred_val),
                    "error_type": error_type,
                    "text_snippet": str(text_val)[:150] if pd.notna(text_val) else "",
                })
    return errors


# ─── Saving and Output ───────────────────────────────────────────────────────

def save_results(all_results: dict, all_errors: dict, metrics_dir: Path, error_dir: Path):
    """Save all evaluation tables to CSV and JSON."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall metrics table
    overall_rows = []
    for model_name, res in all_results.items():
        o = res["overall"]
        overall_rows.append({
            "Model": model_name,
            "Macro_F1": round(o["Macro_F1"], 4),
            "Micro_F1": round(o["Micro_F1"], 4),
            "Micro_Precision": round(o["Micro_Precision"], 4),
            "Micro_Recall": round(o["Micro_Recall"], 4),
            "Accuracy": round(o["Overall_Accuracy"], 4),
            "Exact_Match_Ratio": round(o["Exact_Match_Ratio"], 4),
            "Total_Samples": res["total_samples"],
            "Total_Predicate_Slots": o["Total_Slots"],
        })
    pd.DataFrame(overall_rows).to_csv(metrics_dir / "overall_metrics.csv", index=False)
    print(f"  Saved: {metrics_dir / 'overall_metrics.csv'}")
    
    # 2. Per-predicate metrics
    pred_rows = []
    for model_name, res in all_results.items():
        for pred, m in res["per_predicate"].items():
            pred_rows.append({
                "Model": model_name,
                "Predicate": pred,
                "Predicate_Label": PREDICATE_LABELS[pred],
                "Precision": round(m["Precision"], 4),
                "Recall": round(m["Recall"], 4),
                "F1": round(m["F1"], 4),
                "Accuracy": round(m["Accuracy"], 4),
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
                "Support": m["Support"],
            })
    pd.DataFrame(pred_rows).to_csv(metrics_dir / "per_predicate_metrics.csv", index=False)
    print(f"  Saved: {metrics_dir / 'per_predicate_metrics.csv'}")
    
    # 3. Per-slice metrics
    slice_rows = []
    for model_name, res in all_results.items():
        for slice_name, m in res["per_slice"].items():
            slice_rows.append({
                "Model": model_name,
                "Slice": slice_name,
                "Sample_Count": m["sample_count"],
                "Macro_F1": round(m["Macro_F1"], 4),
                "Micro_F1": round(m["Micro_F1"], 4),
                "Precision": round(m["Micro_Precision"], 4),
                "Recall": round(m["Micro_Recall"], 4),
                "Accuracy": round(m["Accuracy"], 4),
            })
    pd.DataFrame(slice_rows).to_csv(metrics_dir / "per_slice_metrics.csv", index=False)
    print(f"  Saved: {metrics_dir / 'per_slice_metrics.csv'}")
    
    # 4. Extraction errors detail
    all_error_rows = []
    for model_name, errors in all_errors.items():
        for e in errors:
            e_copy = dict(e)
            e_copy["Model"] = model_name
            all_error_rows.append(e_copy)
    pd.DataFrame(all_error_rows).to_csv(error_dir / "extraction_errors_detail.csv", index=False)
    print(f"  Saved: {error_dir / 'extraction_errors_detail.csv'}")
    
    # 5. Error summary by model and predicate
    error_summary_rows = []
    for model_name, errors in all_errors.items():
        error_counts = defaultdict(lambda: {"FP": 0, "FN": 0, "Total": 0})
        for e in errors:
            key = e["predicate"]
            if e["error_type"] == "False Positive":
                error_counts[key]["FP"] += 1
            else:
                error_counts[key]["FN"] += 1
            error_counts[key]["Total"] += 1
        
        for pred in PREDICATES:
            ec = error_counts[pred]
            error_summary_rows.append({
                "Model": model_name,
                "Predicate": PREDICATE_LABELS[pred],
                "False_Positives": ec["FP"],
                "False_Negatives": ec["FN"],
                "Total_Errors": ec["Total"],
            })
    pd.DataFrame(error_summary_rows).to_csv(error_dir / "error_summary.csv", index=False)
    print(f"  Saved: {error_dir / 'error_summary.csv'}")
    
    # 6. Full results JSON
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_results = json.loads(json.dumps(all_results, default=convert_types))
    with open(metrics_dir / "evaluation_results.json", "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {metrics_dir / 'evaluation_results.json'}")


def print_summary(all_results: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("STANDALONE PREDICATE EXTRACTION EVALUATION - SUMMARY")
    print("=" * 80)
    
    print("\n📊 Overall Metrics Comparison:")
    print(f"{'Model':<16} {'Macro-F1':>10} {'Micro-F1':>10} {'Micro-P':>10} {'Micro-R':>10} {'Accuracy':>10}")
    print("-" * 70)
    for model_name, results in all_results.items():
        o = results["overall"]
        print(f"{model_name:<16} {o['Macro_F1']:>10.4f} {o['Micro_F1']:>10.4f} "
              f"{o['Micro_Precision']:>10.4f} {o['Micro_Recall']:>10.4f} "
              f"{o['Overall_Accuracy']:>10.4f}")
    
    print("\n📋 Per-Predicate F1 Comparison:")
    header = f"{'Predicate':<32}"
    for model_name in all_results:
        header += f" {model_name:>15}"
    print(header)
    print("-" * (32 + 16 * len(all_results)))
    
    for pred in PREDICATES:
        row = f"{PREDICATE_LABELS[pred]:<32}"
        for model_name, results in all_results.items():
            f1 = results["per_predicate"][pred]["F1"]
            row += f" {f1:>15.4f}"
        print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Standalone Predicate Extraction")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to ground truth CSV")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to PYTHEN prediction CSVs")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR, help="Output directory for metrics")
    parser.add_argument("--error-dir", type=Path, default=DEFAULT_ERROR_DIR, help="Output directory for error logs")
    args = parser.parse_args()
    
    print("🔬 Running Standalone Predicate Extraction Evaluation")
    print(f"   Ground Truth: {args.ground_truth}")
    print(f"   Predictions:  {args.model_dir}")
    
    gt_df = load_ground_truth(args.ground_truth)
    print(f"📖 Loaded {len(gt_df)} ground truth samples")
    
    all_results = {}
    all_errors = {}
    
    for model_name in MODEL_FILES:
        print(f"\n🔍 Evaluating {model_name}...")
        pred_df = load_model_predictions(model_name, args.model_dir)
        results = evaluate_model(gt_df, pred_df, model_name)
        all_results[model_name] = results
        errors = generate_error_details(gt_df, pred_df, model_name)
        all_errors[model_name] = errors
        print(f"   Done: {len(errors)} extraction errors identified")
    
    print_summary(all_results)
    
    print("\n💾 Saving metrics and error analysis outputs...")
    save_results(all_results, all_errors, args.metrics_dir, args.error_dir)
    print("✅ Extraction evaluation completed successfully!\n")


if __name__ == "__main__":
    main()
