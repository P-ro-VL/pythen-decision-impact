#!/usr/bin/env python3
"""
Step 1: Neuro-Symbolic Error Collection and CoT Reasoning Extraction
====================================================================
Collects all misclassified samples from the Neuro-Symbolic (PYTHEN) pipeline
and cross-references them against Chain-of-Thought (CoT) reasoning steps.
Identifies key divergence cases (where CoT reasoning was intact but symbolic
propagation failed due to predicate hallucination/omission).

Outputs:
- results/error_taxonomy/step1_all_errors.csv
- results/error_taxonomy/step1_gemma_focus_samples.csv
- results/error_taxonomy/step1_summary.csv
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth" / "hearsay_ground_truth.csv"
DEFAULT_PYTHEN_DIR = REPO_ROOT / "data" / "predictions" / "pythen"
DEFAULT_COT_DIR = REPO_ROOT / "data" / "predictions" / "cot"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "error_taxonomy"

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


def normalize_label(val) -> bool:
    """Normalize label to boolean."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return False


def parse_dict(val):
    """Safely parse dict from string."""
    if isinstance(val, dict):
        return val
    s = str(val).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        fixed = s.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        return json.loads(fixed)
    except Exception:
        return {}


def evaluate_pythen_rule(facts: dict) -> bool:
    """
    FRE 801 Symbolic Hearsay Rule Tree:
    R(p) = (is_oral OR is_written OR is_non_verbal) AND is_out_of_court AND is_for_truth
    """
    has_statement = (
        bool(facts.get("is_oral_assertion", False))
        or bool(facts.get("is_written_assertion", False))
        or bool(facts.get("is_non_verbal_assertive_conduct", False))
    )
    return bool(
        has_statement
        and bool(facts.get("is_out_of_court", False))
        and bool(facts.get("is_introduced_to_prove_truth", False))
    )


def load_model_records(model_name: str, pythen_path: Path, cot_path: Path) -> pd.DataFrame:
    """Load and cross-reference PYTHEN outputs and CoT reasoning."""
    df_pythen = pd.read_csv(pythen_path)
    df_cot = pd.read_csv(cot_path) if cot_path.exists() else None

    cot_map = {}
    if df_cot is not None:
        for _, row in df_cot.iterrows():
            idx = int(row["index"])
            thinking = row.get("thinking_process", None)
            if pd.isna(thinking) or not str(thinking).strip():
                parsed = parse_dict(row.get("llm_output", ""))
                thinking = parsed.get("thinking", "")
            cot_map[idx] = thinking

    records = []
    for _, row in df_pythen.iterrows():
        idx = int(row["index"])
        ground_truth = normalize_label(row["answer"])
        text = str(row.get("text", ""))
        slice_cat = str(row.get("slice", ""))

        parsed_ns = parse_dict(row.get("llm_output", ""))
        facts = parsed_ns.get("facts", {}) if isinstance(parsed_ns, dict) else {}
        pred_ns = evaluate_pythen_rule(facts)
        is_ns_correct = (pred_ns == ground_truth)

        raw_cot = cot_map.get(idx, "")
        records.append({
            "model": model_name,
            "index": idx,
            "ground_truth": ground_truth,
            "text": text,
            "slice": slice_cat,
            "pred_ns": pred_ns,
            "is_ns_correct": is_ns_correct,
            "raw_cot_output": str(raw_cot) if raw_cot else "",
            "extracted_json_ns": str(row.get("llm_output", "")),
        })

    return pd.DataFrame(records)


def build_error_list(df: pd.DataFrame) -> pd.DataFrame:
    """Filter error instances where NeSy prediction != ground truth."""
    rows = []
    for _, row in df.iterrows():
        gt = row["ground_truth"]
        pred_ns = row["pred_ns"]
        if pred_ns == gt:
            continue
        has_reasoning = bool(str(row["raw_cot_output"]).strip())
        error_type = "NS_WRONG_WITH_COT" if has_reasoning else "NS_WRONG_NO_COT"
        rows.append({
            "model": row["model"],
            "index": row["index"],
            "ground_truth": gt,
            "pred_ns": pred_ns,
            "error_type": error_type,
            "slice": row["slice"],
            "text": row["text"],
            "raw_cot_output": row["raw_cot_output"],
            "extracted_json_ns": row["extracted_json_ns"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract Neuro-Symbolic Errors")
    parser.add_argument("--pythen-dir", type=Path, default=DEFAULT_PYTHEN_DIR)
    parser.add_argument("--cot-dir", type=Path, default=DEFAULT_COT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP 1: Collecting Neuro-Symbolic Errors & CoT Reasoning")
    print("=" * 70)

    dfs = {}
    all_error_dfs = []

    for model_name, files in MODELS.items():
        pythen_p = args.pythen_dir / files["pythen"]
        cot_p = args.cot_dir / files["cot"]

        df_model = load_model_records(model_name, pythen_p, cot_p)
        dfs[model_name] = df_model
        err_df = build_error_list(df_model)
        all_error_dfs.append(err_df)

        has_cot = (df_model["raw_cot_output"] != "").sum()
        ns_wrong = (df_model["is_ns_correct"] == False).sum()
        print(f"\n[{model_name}]")
        print(f"  Total samples:   {len(df_model)}")
        print(f"  Has CoT context: {has_cot} samples")
        print(f"  NeSy errors:     {ns_wrong} samples")

    combined_errors = pd.concat(all_error_dfs, ignore_index=True)
    print("\n" + "=" * 70)
    print(f"TOTAL: {len(combined_errors)} NeSy errors across all 3 models")
    print("=" * 70)

    gemma_focus = combined_errors[
        (combined_errors["model"] == "Gemma-4-31B") &
        (combined_errors["error_type"] == "NS_WRONG_WITH_COT")
    ].copy()
    gemma_focus["is_focus_sample"] = True
    combined_errors["is_focus_sample"] = (
        (combined_errors["model"] == "Gemma-4-31B") &
        (combined_errors["error_type"] == "NS_WRONG_WITH_COT")
    )

    out_all = args.out_dir / "step1_all_errors.csv"
    out_focus = args.out_dir / "step1_gemma_focus_samples.csv"
    out_summary = args.out_dir / "step1_summary.csv"

    combined_errors.to_csv(out_all, index=False, encoding="utf-8-sig")
    gemma_focus.to_csv(out_focus, index=False, encoding="utf-8-sig")

    summary_rows = []
    for model_name, df_model in dfs.items():
        err_df = combined_errors[combined_errors["model"] == model_name]
        summary_rows.append({
            "model": model_name,
            "total_samples": len(df_model),
            "ns_correct": int((df_model["is_ns_correct"] == True).sum()),
            "ns_wrong": int((df_model["is_ns_correct"] == False).sum()),
            "has_cot_reasoning": int((df_model["raw_cot_output"] != "").sum()),
            "ns_wrong_with_cot": len(err_df[err_df["error_type"] == "NS_WRONG_WITH_COT"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print(f"\n[Saved] Combined errors:  {out_all}")
    print(f"[Saved] Gemma focus set:  {out_focus}")
    print(f"[Saved] Summary table:    {out_summary}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    print("\n✅ Step 1 completed successfully!")


if __name__ == "__main__":
    main()
