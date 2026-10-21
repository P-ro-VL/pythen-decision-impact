# Measuring the Downstream Decision Impact of Predicate Extraction Errors in Symbolic Legal Engines

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference: JURIX 2026](https://img.shields.io/badge/Conference-JURIX%202026-brightgreen.svg)](paper/)
[![Benchmark: LegalBench Hearsay](https://img.shields.io/badge/Benchmark-LegalBench%20(Hearsay)-purple.svg)](data/)

Official repository for the paper:  
**"Measuring the Downstream Decision Impact of Predicate Extraction Errors in Symbolic Legal Engines"**  
*Linh Pham Van, Ha-Thanh Nguyen (VinUniversity), Ken Satoh (Center for Juris-Informatics, ROIS-DS, Japan)*  
Accepted / Targeted for the 39th International Conference on Legal Knowledge and Information Systems (**JURIX 2026**).

---

## 📌 Overview

Neuro-symbolic legal reasoning pipelines couple a Large Language Model (LLM)—which extracts structured legal predicates from unstructured fact patterns—with a deterministic symbolic rule engine that reaches an auditable verdict.

Conventionally, such pipelines are evaluated using standard predicate-level metrics (Precision, Recall, F1-score). However, **aggregate F1 treats every predicate error as equally consequential**—an assumption that fails in legal systems where conditional rule structures let some errors flip the final verdict while absorbing others as redundant:

```
                      ┌────────────────────────────────────────┐
                      │    Unstructured Legal Fact Pattern     │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        LLM Predicate Extractor         │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
          [p_oral, p_written, p_nonverbal, p_out_of_court, p_for_truth]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
       ┌─────────────────────────┐                 ┌─────────────────────────┐
       │   Disjunctive Clause    │                 │    Conjunctive Gates    │
       │ (p_oral ∨ p_written ∨   │                 │   (p_out_of_court ∧     │
       │       p_nonverbal)      │                 │       p_for_truth)      │
       └────────────┬────────────┘                 └────────────┬────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                              ┌───────────────────────┐
                              │  Symbolic Rule Tree   │
                              │     R(p) Verdict      │
                              └───────────────────────┘
```

### Key Contributions & Findings

1. **Structural Criticality (Definition 1):** A closed-form symbolic metric determining whether perturbing a predicate $p_j$ alone flips the verdict $R(p)$ on gold predicate vectors.
2. **Predicate-Level Intervention & Decision Impact Rate (DIR):** We introduce an intervention procedure that corrects one erroneous predicate at a time while holding others fixed, measuring whether the decision flips. DIR quantifies the empirical criticality of each predicate.
3. **Extraction Difficulty $\neq$ Decision Importance:**
   - **`is_written_assertion`**: Hardest to extract ($F_1 \in [0.286, 0.429]$) and highly critical ($\text{DIR} \in [0.700, 0.900]$).
   - **`is_non_verbal_assertive_conduct`**: Comparably hard ($F_1 \in [0.593, 0.667]$) but almost entirely inconsequential ($\text{DIR} \in [0.000, 0.091]$) because other assertion predicates satisfy the disjunction.
   - **Conjunctive Gates (`is_out_of_court`, `is_introduced_to_prove_truth`)**: Easy to extract ($F_1 > 0.86$ for Gemma), but errors carry near-fatal risk ($\text{DIR}$ up to $1.000$) due to lack of a disjunctive safety net.
4. **Model-Specific Risk Profiles:** Hallucinated predicates (E1) are critical in only $46.9\% - 59.5\%$ of instances for GPT-OSS 120B and DeepSeek V3.2, but reach **$96.2\%$ criticality for Gemma 4 31B**, demonstrating that predicate F1 rankings and downstream decision reliability rankings do not necessarily coincide.

---

## 📊 Summary Results

### 1. End-to-End Pipeline Comparison ($N=94$ Samples)

| Pipeline Configuration | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|:---:|:---:|:---:|:---:|
| **Naive Zero-Shot LLM** | DeepSeek V3.2 | 0.6915 | 0.6034 | 0.8537 | 0.7071 |
| | GPT-OSS 120B | 0.6489 | 0.6667 | 0.3902 | 0.4923 |
| | Gemma 4 31B | 0.7340 | 0.6739 | 0.7561 | 0.7126 |
| **Chain-of-Thought (CoT)** | DeepSeek V3.2 | 0.7447 | 0.6349 | **0.9756** | 0.7692 |
| | GPT-OSS 120B | 0.7660 | 0.7436 | 0.7073 | 0.7250 |
| | Gemma 4 31B | 0.7766 | 0.6667 | **0.9756** | 0.7921 |
| **Neuro-Symbolic (PYTHEN / Ours)** | DeepSeek V3.2 | **0.7766** | **0.6786** | 0.9268 | **0.7835** |
| | GPT-OSS 120B | **0.8191** | **0.7400** | 0.9024 | **0.8132** |
| | Gemma 4 31B | **0.8404** | **0.7500** | 0.9512 | **0.8387** |

*Oracle Experiment:* Feeding gold predicates directly into the PYTHEN rule tree yields $\text{Accuracy} = \text{Precision} = \text{Recall} = \text{F1} = 1.000$, confirming that symbolic inference is sound and all residual errors originate in predicate extraction.

### 2. Extraction F1 vs. Decision Impact Rate (DIR)

| Predicate | DeepSeek F1 | DeepSeek DIR | GPT-OSS F1 | GPT-OSS DIR | Gemma F1 | Gemma DIR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `is_oral_assertion` | 0.8972 | 0.1818 | 0.9286 | 0.1250 | **0.9565** | 0.2000 |
| `is_written_assertion` | 0.2857 | **0.7000** | 0.4286 | **0.8750** | 0.3750 | **0.9000** |
| `is_non_verbal_assertive_conduct` | 0.5926 | 0.0909 | 0.6667 | **0.0000** | 0.6667 | **0.0000** |
| `is_out_of_court` | 0.9104 | 0.1667 | 0.8769 | 0.1875 | **0.9760** | **1.0000** |
| `is_introduced_to_prove_truth` | 0.7603 | **0.6207** | 0.7442 | 0.3939 | 0.8644 | **0.8750** |

---

## 📁 Repository Structure

```
.
├── README.md                          # Project documentation and reproduction guide
├── LICENSE                            # MIT License
├── requirements.txt                   # Lightweight python dependencies
├── .gitignore                         # Standard git ignore rules (Python, OS, LaTeX)
│
├── data/                              # Benchmark data and model predictions
│   ├── ground_truth/
│   │   ├── hearsay_ground_truth.csv   # 94 LegalBench samples with 5 annotated FRE 801 predicates
│   │   └── hearsay_ground_truth.json  # JSON schema representation with metadata & proffer
│   └── predictions/
│       ├── naive/                     # Direct zero-shot LLM predictions
│       │   ├── naive_deepseek_v3.2.csv
│       │   ├── naive_gpt_oss_120b.csv
│       │   └── naive_gemma_4_31b.csv
│       ├── cot/                       # Chain-of-Thought (CoT) baseline outputs & reasoning traces
│       │   ├── cot_deepseek_v3.2.csv
│       │   ├── cot_gpt_oss_120b.csv
│       │   └── cot_gemma_4_31b.csv
│       └── pythen/                    # Neuro-symbolic (LLM predicate extraction) outputs
│           ├── pythen_deepseek_v3.2.csv
│           ├── pythen_gpt_oss_120b.csv
│           └── pythen_gemma_4_31b.csv
│
├── src/                               # Reproduction & evaluation source code
│   ├── run_all.py                     # Master script reproducing all paper tables
│   ├── evaluate_extraction.py         # Standalone predicate extraction evaluation (Macro/Micro F1)
│   ├── step1_error_extraction.py      # NeSy error collection & CoT reasoning cross-referencing
│   └── step2_error_taxonomy.py        # Error taxonomy (E1-E5), intervention & DIR computation
│
├── results/                           # Experimental outputs, metrics, and companion reports
│   ├── summary_metrics/
│   │   ├── pipeline_comparison.csv    # End-to-end performance across all 3 pipelines
│   │   ├── overall_metrics.csv        # Extraction macro/micro precision, recall, F1, accuracy
│   │   ├── per_predicate_metrics.csv  # Predicate-level F1, TP, FP, FN, TN
│   │   ├── per_slice_metrics.csv      # Slice-based breakdown across hearsay categories
│   │   └── evaluation_results.json    # Complete serialized metrics
│   ├── error_taxonomy/
│   │   ├── step2_predicate_accuracy.csv  # DIR and criticality breakdown per predicate
│   │   ├── step2_error_taxonomy_full.csv # Full sample-level error taxonomy annotations
│   │   ├── step2_error_taxonomy_llm.csv  # Extraction tier errors (E1, E2, E3)
│   │   ├── step2_error_taxonomy_cot.csv  # Reasoning tier errors (E4, E5)
│   │   └── step2_annotation_sheet.csv    # Diagnostic annotation sheet with fact patterns
│   └── reports/
│       ├── technical_report.md        # Comprehensive English technical report
│       ├── hearsay_error_analysis.md  # Detailed qualitative hearsay analysis
│       ├── hearsay_ablation_study_report.md # Ablation study report
│       └── predicate_extraction_report.md   # Standalone linguistic extraction report
│
├── paper/                             # LaTeX manuscripts and compiled camera-ready documents
│   ├── jurix_full/                    # 15-page extended version (LaTeX source + main.pdf)
│   └── jurix_short/                   # JURIX conference edition (LaTeX source + main.pdf)
│
└── docs/                              # Research documentation & roadmap
    └── notes/                         # Background research notes and journal expansion plan
        ├── paper_idea.md
        ├── journal_extension_roadmap.md
        └── jurix_purpose_and_idea.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/<your-username>/PYTHEN_PAPER_GITHUB.git
cd PYTHEN_PAPER_GITHUB

# Recommended: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (pandas, numpy)
pip install -r requirements.txt
```

### 2. One-Click Full Reproduction

To reproduce all tables, error taxonomies, and Decision Impact Rates in the paper:

```bash
python src/run_all.py
```

### 3. Running Individual Experiments

- **Standalone Predicate Extraction Evaluation (Table 2 & 3 in paper):**
  ```bash
  python src/evaluate_extraction.py
  ```
  *Computes Per-predicate F1, Macro-F1, Micro-F1, exact-match ratios, and slice breakdowns.*

- **Error Collection & CoT Divergence Analysis (Step 1):**
  ```bash
  python src/step1_error_extraction.py
  ```
  *Cross-references NeSy errors against CoT thinking traces.*

- **Error Taxonomy & Decision Impact Rate (DIR) (Table 4 & 5 in paper):**
  ```bash
  python src/step2_error_taxonomy.py
  ```
  *Executes predicate-level interventions to compute structural criticality and DIR.*

---

## 📚 Dataset Details

The experiments are conducted on the `hearsay` subset of **LegalBench**, comprising 94 manually validated legal fact patterns. For each fact pattern, 5 binary legal predicates are evaluated under Federal Rules of Evidence (FRE) 801:

1. `is_oral_assertion`: Spoken verbal statement.
2. `is_written_assertion`: Document, text, or written communication offered as an assertion.
3. `is_non_verbal_assertive_conduct`: Conduct intended by the actor as an assertion (e.g., pointing).
4. `is_out_of_court`: Statement made outside of the current trial or hearing.
5. `is_introduced_to_prove_truth`: Evidence offered to prove the exact truth of what it asserts.

The symbolic rule evaluates:
$$\mathcal{R}(p) = (p_{\text{oral}} \lor p_{\text{written}} \lor p_{\text{nonverbal}}) \land p_{\text{out-of-court}} \land p_{\text{truth}}$$

---

## 📖 Citation

If you find this repository or paper useful in your research, please cite:

```bibtex
@inproceedings{pham2026measuring,
  title={Measuring the Downstream Decision Impact of Predicate Extraction Errors in Symbolic Legal Engines},
  author={Pham Van, Linh and Nguyen, Ha-Thanh and Satoh, Ken},
  booktitle={Proceedings of the 39th International Conference on Legal Knowledge and Information Systems (JURIX 2026)},
  year={2026},
  publisher={IOS Press}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
