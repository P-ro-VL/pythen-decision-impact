# Ablation Study Report: Separating LLM Predicate Extraction from PYTHEN Reasoning

## Executive Summary

This report follows the requested ablation design, using the current files under `dataset/`. Config C (`LLM Extract + LLM Logic`) is intentionally omitted. GPT OSS Neuro-Symbolic now contains 94 rows; the previously missing samples were added with all predicates set to `false`.

- Config A maps to `dataset/preliminary_experiment_results/baseline/`.
- Config B maps to `dataset/preliminary_experiment_results/naive/`, interpreted as the LLM-only text-rule setting available in the experiment folder.
- Config D maps to `dataset/preliminary_experiment_results/pythen/`.
- Oracle uses `dataset/ground_truth_predicates.json`, generated from `dataset/hearsay_predicates_breakdown.csv`.

## Step 1. Oracle Predicates Dataset

Created or refreshed gold predicate file: `dataset/ground_truth_predicates.json`. It contains 94 samples with the five ground-truth predicates, the ground-truth hearsay label, slice, and fact pattern.

Schema used:

```json
{
  "sample_id": "hearsay_01",
  "index": 1,
  "ground_truth_hearsay": true,
  "answer_label": "Yes",
  "slice": "example slice",
  "fact_pattern": "...",
  "ground_truth_facts": {
    "is_oral_assertion": true,
    "is_written_assertion": false,
    "is_non_verbal_assertive_conduct": false,
    "is_out_of_court": true,
    "is_introduced_to_prove_truth": true
  }
}
```

## Step 2. Oracle Experiment: GT Predicates + PYTHEN

Oracle evaluation feeds the gold predicates directly into the hearsay Boolean rule, bypassing all LLM extraction.

| N | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 94 | 1.000 | 1.000 | 1.000 | 1.000 | 41 | 53 | 0 | 0 |

Oracle result is 100%, showing that the current hearsay rule tree is logically aligned with the gold predicate labels. The remaining gap in Config D comes from predicate extraction, not symbolic inference.

## Step 3. Standalone Predicate Extraction Evaluation

The table below compares each model's extracted JSON facts against the gold predicate table. GPT OSS has 94 available JSON outputs after inserting the formerly missing samples as all-false predicate vectors.

| Model | Available Samples | Predicate Slots | Micro Precision | Micro Recall | Micro F1 | Macro F1 | Slot Accuracy | TP | FP | FN |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek V4 Flash | 94 | 470 | 0.789 | 0.851 | 0.819 | 0.689 | 0.845 | 165 | 44 | 29 |
| GPT OSS 20B/120B | 94 | 470 | 0.775 | 0.871 | 0.820 | 0.729 | 0.843 | 169 | 49 | 25 |
| Gemma 4 31B | 94 | 470 | 0.873 | 0.918 | 0.894 | 0.768 | 0.911 | 178 | 26 | 16 |

### Per-Predicate F1

| Predicate | DeepSeek V4 Flash | GPT OSS 20B/120B | Gemma 4 31B |
|--|--:|--:|--:|
| `is_oral_assertion` | 0.897 | 0.929 | 0.957 |
| `is_written_assertion` | 0.286 | 0.429 | 0.375 |
| `is_non_verbal_assertive_conduct` | 0.593 | 0.667 | 0.667 |
| `is_out_of_court` | 0.910 | 0.877 | 0.976 |
| `is_introduced_to_prove_truth` | 0.760 | 0.744 | 0.864 |

## Step 4. Component Matrix

Config C is omitted by design. All reported values are computed from the current experiment outputs.

| Model | Config A Baseline CoT F1 | Config B LLM + Text Rules F1 | Config D Ours F1 | Oracle GT + PYTHEN F1 | D - A | D - B | Oracle - D |
|--|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek V4 Flash | 0.769 | 0.707 | 0.784 | 1.000 | 0.014 | 0.076 | 0.216 |
| GPT OSS 20B/120B | 0.725 | 0.492 | 0.813 | 1.000 | 0.088 | 0.321 | 0.187 |
| Gemma 4 31B | 0.792 | 0.713 | 0.839 | 1.000 | 0.047 | 0.126 | 0.161 |

### Full Classification Metrics by Configuration

| Model | Config | N | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek V4 Flash | A: Baseline CoT | 94 | 0.745 | 0.635 | 0.976 | 0.769 | 40 | 30 | 23 | 1 |
| DeepSeek V4 Flash | B: LLM + Text Rules | 94 | 0.691 | 0.603 | 0.854 | 0.707 | 35 | 30 | 23 | 6 |
| DeepSeek V4 Flash | D: Ours (LLM + PYTHEN) | 94 | 0.777 | 0.679 | 0.927 | 0.784 | 38 | 35 | 18 | 3 |
| GPT OSS 20B/120B | A: Baseline CoT | 94 | 0.766 | 0.744 | 0.707 | 0.725 | 29 | 43 | 10 | 12 |
| GPT OSS 20B/120B | B: LLM + Text Rules | 94 | 0.649 | 0.667 | 0.390 | 0.492 | 16 | 45 | 8 | 25 |
| GPT OSS 20B/120B | D: Ours (LLM + PYTHEN) | 94 | 0.819 | 0.740 | 0.902 | 0.813 | 37 | 40 | 13 | 4 |
| Gemma 4 31B | A: Baseline CoT | 94 | 0.777 | 0.667 | 0.976 | 0.792 | 40 | 33 | 20 | 1 |
| Gemma 4 31B | B: LLM + Text Rules | 94 | 0.734 | 0.674 | 0.756 | 0.713 | 31 | 38 | 15 | 10 |
| Gemma 4 31B | D: Ours (LLM + PYTHEN) | 94 | 0.840 | 0.750 | 0.951 | 0.839 | 39 | 40 | 13 | 2 |
| Oracle | Oracle: GT + PYTHEN | 94 | 1.000 | 1.000 | 1.000 | 1.000 | 41 | 53 | 0 | 0 |

## Step 5. Ablation Table and Contribution Analysis

| Model | Accuracy A | Accuracy B | Accuracy D | Oracle Accuracy | D - A | Oracle - D |
|--|--:|--:|--:|--:|--:|--:|
| DeepSeek V4 Flash | 0.745 | 0.691 | 0.777 | 1.000 | 0.032 | 0.223 |
| GPT OSS 20B/120B | 0.766 | 0.649 | 0.819 | 1.000 | 0.053 | 0.181 |
| Gemma 4 31B | 0.777 | 0.734 | 0.840 | 1.000 | 0.064 | 0.160 |

### Key Takeaways

- DeepSeek V4 Flash: Config D reaches F1=0.784, compared with Config A F1=0.769 and Config B F1=0.707. The remaining Oracle-D gap is 0.216 F1, while predicate micro-F1 is 0.819.
- GPT OSS 20B/120B: Config D reaches F1=0.813, compared with Config A F1=0.725 and Config B F1=0.492. The remaining Oracle-D gap is 0.187 F1, while predicate micro-F1 is 0.820.
- Gemma 4 31B: Config D reaches F1=0.839, compared with Config A F1=0.792 and Config B F1=0.713. The remaining Oracle-D gap is 0.161 F1, while predicate micro-F1 is 0.894.
- Because the Oracle setting reaches perfect performance, PYTHEN itself does not introduce logical error under the current gold predicates. The non-oracle errors in Config D are attributable to imperfect predicate extraction.
- The largest practical diagnostic signal is the gap between Config D and Oracle: it directly estimates how much performance can still be recovered by improving the LLM extraction layer while keeping the symbolic rule tree fixed.
- Config B shows whether presenting legal rules in text helps the LLM without symbolic execution. In the current data, Config D exceeds Config B for all three models, supporting deterministic rule execution over relying on the LLM to operationalize the rule in free-form reasoning.

## Data Caveats

- Ground-truth CSV indices contain 94 samples with IDs `0` to `94`, but index `93` is absent in the current files. The report follows the actual 94-row ground-truth index set.
- No configuration has missing rows relative to the 94-sample ground-truth index set.
- GPT OSS Neuro-Symbolic rows `[8, 46, 51, 65, 87]` are synthetic all-false predicate rows added to make the file complete at 94 samples.
