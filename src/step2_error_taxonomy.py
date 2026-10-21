#!/usr/bin/env python3
"""
Step 2: Error Taxonomy, Decision Impact Rate (DIR) & Structural Criticality
===========================================================================
Implements the core methodological contributions of the paper:
1. Extraction Error Taxonomy:
   - E1: Predicate Hallucination (False Positive)
   - E2: Predicate Omission (False Negative)
   - E3: Invalid / Missing Reference Text
2. Chain-of-Thought (CoT) Error Taxonomy:
   - E4: Conservatism Bias (False Negative despite valid predicate steps)
   - E5: Logical Bypass in CoT (False Positive jumping over legal elements)
3. Predicate-level Intervention & Decision Impact Rate (DIR):
   - Definition 1: Structural Criticality (closed-form on gold vector)
   - Empirical DIR: fraction of errors that actually flip the final verdict
   - Refined Taxonomy: Critical vs. Benign breakdown

Outputs:
- results/error_taxonomy/step2_error_taxonomy_full.csv
- results/error_taxonomy/step2_error_taxonomy_llm.csv
- results/error_taxonomy/step2_error_taxonomy_cot.csv
- results/error_taxonomy/step2_annotation_sheet.csv
- results/error_taxonomy/step2_predicate_accuracy.csv
"""

import argparse
import ast
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth" / "hearsay_ground_truth.csv"
DEFAULT_PYTHEN_DIR = REPO_ROOT / "data" / "predictions" / "pythen"
DEFAULT_COT_DIR = REPO_ROOT / "data" / "predictions" / "cot"
DEFAULT_STEP1_ERRORS = REPO_ROOT / "results" / "error_taxonomy" / "step1_all_errors.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "error_taxonomy"

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

MODELS = {
    "DeepSeek-V3.2": {
        "pythen": "pythen_deepseek_v3.2.csv",
        "cot": "cot_deepseek_v3.2.csv",
    },
    "GPT-OSS-120B": {
        "pythen": "pythen_gpt_oss_120b.csv",
        "cot": "cot_gpt_oss_120b.csv",
    },
    "Gemma-4-31B": {
        "pythen": "pythen_gemma_4_31b.csv",
        "cot": "cot_gemma_4_31b.csv",
    },
}


def bool_val(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().upper()
    return s in ("TRUE", "YES", "1")


def parse_dict(raw):
    if pd.isna(raw) or not isinstance(raw, (str, dict)):
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        fixed = s.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        return json.loads(fixed)
    except Exception:
        return {}


def evaluate_pythen_rule(pred_dict: dict) -> bool:
    """Evaluate symbolic rule R(p) for hearsay."""
    assertion = (
        bool_val(pred_dict.get("is_oral_assertion", False))
        or bool_val(pred_dict.get("is_written_assertion", False))
        or bool_val(pred_dict.get("is_non_verbal_assertive_conduct", False))
    )
    return bool(
        assertion
        and bool_val(pred_dict.get("is_out_of_court", False))
        and bool_val(pred_dict.get("is_introduced_to_prove_truth", False))
    )


def compute_structural_criticality(gold_p: dict) -> dict:
    """
    Definition 1 (Structural Criticality):
    p_j is structurally critical if flipping p_j alone alters R(p).
    """
    base_verdict = evaluate_pythen_rule(gold_p)
    criticality = {}
    for pred in PREDICATES:
        perturbed = dict(gold_p)
        perturbed[pred] = not perturbed[pred]
        perturbed_verdict = evaluate_pythen_rule(perturbed)
        criticality[pred] = (perturbed_verdict != base_verdict)
    return criticality


def compute_decision_impact(model_preds: dict, gold_preds: dict) -> dict:
    """
    Predicate-level intervention (Eq. 4-5):
    For each erroneous predicate j, flip p_j back to gold while keeping
    other erroneous predicates unchanged. Measure if decision flips.
    """
    base_verdict = evaluate_pythen_rule(model_preds)
    impacts = {}
    for pred in PREDICATES:
        if model_preds[pred] != gold_preds[pred]:
            # Intervene: correct this single predicate
            intervened = dict(model_preds)
            intervened[pred] = gold_preds[pred]
            intervened_verdict = evaluate_pythen_rule(intervened)
            impacts[pred] = (intervened_verdict != base_verdict)
        else:
            impacts[pred] = None  # No error on this predicate
    return impacts


def main():
    parser = argparse.ArgumentParser(description="Step 2: Error Taxonomy & Decision Impact Rate")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--pythen-dir", type=Path, default=DEFAULT_PYTHEN_DIR)
    parser.add_argument("--cot-dir", type=Path, default=DEFAULT_COT_DIR)
    parser.add_argument("--step1-errors", type=Path, default=DEFAULT_STEP1_ERRORS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print("STEP 2: Error Taxonomy, Decision Impact Rate (DIR) & Criticality")
    print("=" * 75)

    # 1. Load Ground Truth
    gt_df = pd.read_csv(args.ground_truth)
    gt_df["index"] = gt_df["index"].astype(int)
    for p in PREDICATES:
        gt_df[p] = gt_df[p].apply(bool_val)
    gt_df["ground_truth_hearsay"] = gt_df["ground_truth"].apply(bool_val)
    print(f"📖 Ground truth loaded: {len(gt_df)} samples")

    # 2. Load Models
    pythen_dfs = {}
    baseline_dfs = {}
    for model_name, files in MODELS.items():
        pdf = pd.read_csv(args.pythen_dir / files["pythen"])
        pdf["index"] = pdf["index"].astype(int)
        pythen_dfs[model_name] = pdf

        cpath = args.cot_dir / files["cot"]
        if cpath.exists():
            cdf = pd.read_csv(cpath)
            cdf["index"] = cdf["index"].astype(int)
            baseline_dfs[model_name] = cdf

    # 3. Phase 1: LLM Extraction Taxonomy & Decision Impact
    print("\n🔍 Phase 1: Computing Extraction Taxonomy & Decision Impact (DIR)...")
    taxonomy_rows = []
    dir_counts = defaultdict(lambda: defaultdict(lambda: {"errors": 0, "critical": 0}))

    for model_name, pdf in pythen_dfs.items():
        merged = pdf.merge(gt_df[["index"] + PREDICATES + ["ground_truth_hearsay", "text", "slice"]], on="index")
        for _, row in merged.iterrows():
            idx = int(row["index"])
            parsed = parse_dict(row["llm_output"])
            facts = parsed.get("facts", {})
            refs = parsed.get("reference_texts", {})

            model_p = {p: bool_val(facts.get(p, False)) for p in PREDICATES}
            gold_p = {p: bool_val(row[p]) for p in PREDICATES}

            error_codes = []
            pred_errors = {}
            impacts = compute_decision_impact(model_p, gold_p)

            for p in PREDICATES:
                p_err = []
                if model_p[p] and not gold_p[p]:
                    p_err.append("E1")
                    error_codes.append("E1")
                    dir_counts[model_name][p]["errors"] += 1
                    if impacts[p]:
                        dir_counts[model_name][p]["critical"] += 1
                elif not model_p[p] and gold_p[p]:
                    p_err.append("E2")
                    error_codes.append("E2")
                    dir_counts[model_name][p]["errors"] += 1
                    if impacts[p]:
                        dir_counts[model_name][p]["critical"] += 1
                elif model_p[p] and gold_p[p]:
                    ref = refs.get(p, None) if isinstance(refs, dict) else None
                    if ref is None or (isinstance(ref, str) and not ref.strip()):
                        p_err.append("E3")
                        error_codes.append("E3")

                pred_errors[p] = {
                    "errors": p_err,
                    "is_critical": impacts[p],
                }

            if not error_codes:
                error_codes = ["CORRECT"]

            taxonomy_rows.append({
                "model": model_name,
                "index": idx,
                "tier": "LLM_Extraction",
                "error_codes": "|".join(sorted(set(error_codes))),
                "predicate_errors": json.dumps(pred_errors),
                "ground_truth_hearsay": row["ground_truth_hearsay"],
                "slice": row.get("slice", ""),
                "text": str(row.get("text", ""))[:200],
            })

    # 4. Phase 2: Baseline CoT Taxonomy
    print("🔍 Phase 2: Computing Baseline CoT Taxonomy (E4/E5)...")
    for model_name, cdf in baseline_dfs.items():
        merged = cdf.merge(gt_df[["index", "ground_truth_hearsay", "text", "slice"]], on="index")
        for _, row in merged.iterrows():
            idx = int(row["index"])
            gt_hearsay = row["ground_truth_hearsay"]
            pred_hearsay = bool_val(row.get("prediction", False))

            if pred_hearsay == gt_hearsay:
                continue

            parsed_out = parse_dict(row.get("llm_output", ""))
            thinking = parsed_out.get("thinking", {}) if isinstance(parsed_out, dict) else {}
            if not thinking and "thinking_process" in row:
                thinking = parse_dict(row["thinking_process"])

            step1_text = str(thinking.get("step1", "")).lower()
            step2_text = str(thinking.get("step2", "")).lower()
            step3_text = str(thinking.get("step3", "")).lower()
            conclusion = str(thinking.get("conclusion", "")).lower()

            error_codes = []
            if gt_hearsay and not pred_hearsay:
                # Ground truth hearsay, but CoT said False -> E4 Conservatism or E5 Logical jump
                if "not" in step3_text or "not hearsay" in conclusion:
                    error_codes.append("E4")
                else:
                    error_codes.append("E5")
            elif not gt_hearsay and pred_hearsay:
                error_codes.append("E5")

            if not error_codes:
                error_codes = ["E_OTHER"]

            taxonomy_rows.append({
                "model": model_name,
                "index": idx,
                "tier": "Baseline_CoT",
                "error_codes": "|".join(error_codes),
                "predicate_errors": "{}",
                "ground_truth_hearsay": gt_hearsay,
                "slice": row.get("slice", ""),
                "text": str(row.get("text", ""))[:200],
            })

    taxonomy_df = pd.DataFrame(taxonomy_rows)

    # 5. Phase 3: Detailed Annotation Sheet
    print("🔍 Phase 3: Building Annotation Sheet...")
    annotation_rows = []
    for model_name, pdf in pythen_dfs.items():
        merged = pdf.merge(gt_df[["index"] + PREDICATES + ["ground_truth_hearsay", "text", "slice"]], on="index")
        for _, row in merged.iterrows():
            idx = int(row["index"])
            parsed = parse_dict(row["llm_output"])
            facts = parsed.get("facts", {})
            model_p = {p: bool_val(facts.get(p, False)) for p in PREDICATES}
            gold_p = {p: bool_val(row[p]) for p in PREDICATES}

            pred_verdict = evaluate_pythen_rule(model_p)
            gt_verdict = row["ground_truth_hearsay"]

            if pred_verdict == gt_verdict:
                continue

            error_details = []
            impacts = compute_decision_impact(model_p, gold_p)
            critical_pred = ""

            for p in PREDICATES:
                if model_p[p] and not gold_p[p]:
                    is_crit = impacts[p]
                    tag = "Critical" if is_crit else "Benign"
                    error_details.append(f"E1({tag}): {p}")
                    if is_crit and not critical_pred:
                        critical_pred = p
                elif not model_p[p] and gold_p[p]:
                    is_crit = impacts[p]
                    tag = "Critical" if is_crit else "Benign"
                    error_details.append(f"E2({tag}): {p}")
                    if is_crit and not critical_pred:
                        critical_pred = p

            annotation_rows.append({
                "sample_id": f"#{idx}",
                "model": model_name,
                "index": idx,
                "slice": row.get("slice", ""),
                "fact_pattern": str(row.get("text", ""))[:250],
                "ground_truth_hearsay": gt_verdict,
                "model_prediction": pred_verdict,
                "critical_predicate": critical_pred,
                "error_details": " | ".join(error_details) if error_details else "Rule absorbed",
                "oral_gt": gold_p["is_oral_assertion"],
                "written_gt": gold_p["is_written_assertion"],
                "nonverbal_gt": gold_p["is_non_verbal_assertive_conduct"],
                "out_of_court_gt": gold_p["is_out_of_court"],
                "for_truth_gt": gold_p["is_introduced_to_prove_truth"],
                "oral_pred": model_p["is_oral_assertion"],
                "written_pred": model_p["is_written_assertion"],
                "nonverbal_pred": model_p["is_non_verbal_assertive_conduct"],
                "out_of_court_pred": model_p["is_out_of_court"],
                "for_truth_pred": model_p["is_introduced_to_prove_truth"],
            })

    annotation_df = pd.DataFrame(annotation_rows)

    # 6. Phase 4: Predicate Accuracy Table
    accuracy_rows = []
    for model_name, pdf in pythen_dfs.items():
        merged = pdf.merge(gt_df[["index"] + PREDICATES], on="index")
        for p in PREDICATES:
            parsed_preds = [bool_val(parse_dict(raw).get("facts", {}).get(p, False)) for raw in merged["llm_output"]]
            gt_vals = merged[p].tolist()
            tp = sum(1 for y, y_hat in zip(gt_vals, parsed_preds) if y and y_hat)
            fp = sum(1 for y, y_hat in zip(gt_vals, parsed_preds) if not y and y_hat)
            fn = sum(1 for y, y_hat in zip(gt_vals, parsed_preds) if y and not y_hat)
            tn = sum(1 for y, y_hat in zip(gt_vals, parsed_preds) if not y and not y_hat)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            dir_val = (
                dir_counts[model_name][p]["critical"] / dir_counts[model_name][p]["errors"]
                if dir_counts[model_name][p]["errors"] > 0
                else 0.0
            )

            accuracy_rows.append({
                "model": model_name,
                "predicate": p,
                "predicate_label": PREDICATE_LABELS[p],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "total_errors": fp + fn,
                "critical_errors": dir_counts[model_name][p]["critical"],
                "DIR": round(dir_val, 4),
            })
    accuracy_df = pd.DataFrame(accuracy_rows)

    # 7. Save outputs
    print("\n💾 Saving taxonomy outputs...")
    llm_tax = taxonomy_df[taxonomy_df["tier"] == "LLM_Extraction"]
    cot_tax = taxonomy_df[taxonomy_df["tier"] == "Baseline_CoT"]

    llm_tax.to_csv(args.out_dir / "step2_error_taxonomy_llm.csv", index=False)
    cot_tax.to_csv(args.out_dir / "step2_error_taxonomy_cot.csv", index=False)
    taxonomy_df.to_csv(args.out_dir / "step2_error_taxonomy_full.csv", index=False)
    annotation_df.to_csv(args.out_dir / "step2_annotation_sheet.csv", index=False)
    accuracy_df.to_csv(args.out_dir / "step2_predicate_accuracy.csv", index=False)

    print(f"  Saved: {args.out_dir / 'step2_error_taxonomy_llm.csv'}")
    print(f"  Saved: {args.out_dir / 'step2_error_taxonomy_cot.csv'}")
    print(f"  Saved: {args.out_dir / 'step2_error_taxonomy_full.csv'}")
    print(f"  Saved: {args.out_dir / 'step2_annotation_sheet.csv'}")
    print(f"  Saved: {args.out_dir / 'step2_predicate_accuracy.csv'}")

    # Summary table
    print("\n" + "=" * 75)
    print("DECISION IMPACT RATE (DIR) & EXTRACTION F1 SUMMARY")
    print("=" * 75)
    print(f"{'Model':<16} {'Predicate':<32} {'F1':>8} {'Errors':>8} {'Critical':>10} {'DIR':>8}")
    print("-" * 84)
    for _, r in accuracy_df.iterrows():
        print(f"{r['model']:<16} {r['predicate_label']:<32} {r['f1']:>8.4f} "
              f"{r['total_errors']:>8} {r['critical_errors']:>10} {r['DIR']:>8.4f}")
    print("=" * 75)
    print("\n✅ Step 2 completed successfully!")


if __name__ == "__main__":
    main()
