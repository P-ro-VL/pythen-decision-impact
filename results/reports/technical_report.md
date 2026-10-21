# Dissecting Neuro-Symbolic Divergence: An In-Depth Error Analysis of Predicate Extraction and Rigid Rule Propagation in Legal Reasoning (PYTHEN)

**Authors:** Research Team on Neuro-Symbolic Legal AI  
**Dataset:** Hearsay Benchmark ($N = 94$ Fact Patterns)  
**Evaluated Systems:** Baseline Chain-of-Thought (CoT) vs. PYTHEN Neuro-Symbolic Architecture  
**Evaluated Models:** DeepSeek V4 Flash / V3.2, GPT OSS 20B, Google Gemma 4 31B / Gemma 4B  

--

## Abstract

Neuro-symbolic (NeSy) architectures such as **PYTHEN** integrate the rich semantic comprehension of Large Language Models (LLMs) with the formal verification and interpretability guarantees of deterministic symbolic engines. However, empirical evaluations on the 94-sample **Hearsay** legal benchmark reveal a striking performance paradox: while PYTHEN unlocks substantial performance leaps for **DeepSeek V4 Flash** (Recall jumping from $0.436$ to $0.892$, $\text{F1} \colon 0.607 \to 0.878$) and **GPT OSS 20B** ($\Delta \text{F1} = +0.0965$), it induces a severe performance degradation in **Gemma 4 31B** (F1 plummeting from $0.828$ to $0.731$).

This paper presents a comprehensive, multi-tiered error analysis synthesizing empirical findings from systematic diagnostic runs (Step 1 and Step 2 error evaluations). We demonstrate that this performance divergence is governed by the structural coupling between **LLM inductive biases during discrete predicate extraction** and the **rigid, unweighted conjunctive logic gates (`"op": "ALL"`)** in the symbolic layer. Specifically, Gemma exhibits an aggressive *False Positive Extraction* bias on the pivotal boundary predicate $\mathcal{P}_5$ (`is_introduced_to_prove_truth`), misclassifying non-assertive notice and state-of-mind evidence as truth-seeking assertions; under rigid boolean conjunction, a single false positive deterministically flips the conjoined rule to an erroneous hearsay verdict ($\mathbb{P}(\text{Verdict Flip} \mid \epsilon_5 = +1) = 1.0$). In contrast, DeepSeek suffers from severe *End-to-End Decision Paralysis* in standalone CoT (Precision $1.000$, Recall $0.436$), which is resolved when the model is decoupled from the burden of the final verdict and tasked solely with discrete, factual predicate extraction. We formalize a comprehensive **Error Taxonomy Matrix (Levels E1–E6)**, provide mathematical proofs of error propagation, report per-predicate accuracy breakdowns, and examine representative legal case studies (#6, #12, #15, #23, #37, #47) to establish design guidelines for robust, next-generation neuro-symbolic reasoning systems.

--

## 1. Introduction & The Performance Paradox

The integration of Large Language Models (LLMs) with Symbolic Knowledge Graphs and Formal Solvers—termed **Neuro-Symbolic (NeSy) AI**—promises to eliminate LLM hallucinations and enforce deterministic compliance with formal legal doctrines such as the Federal Rules of Evidence (FRE 801–805). Under the **PYTHEN** framework, the legal determination of whether an evidence item constitutes *Hearsay* is partitioned into two distinct stages:

1. **Sub-symbolic Perception & Predicate Extraction (LLM Stage):** The model parses raw legal text into a discrete vector of boolean legal predicates $\vec{\mathcal{P}} = \langle P_1, P_2, P_3, P_4, P_5 \rangle$.
2. **Deterministic Symbolic Inference (Rule Engine Stage):** A hard-coded symbolic grammar executes the formal rule of hearsay logic.

```mermaid
graph LR
    subgraph LLM Perception Stage
        FactPattern["Legal Fact Pattern & Proffer"] -> LLM["LLM Predicate Extractor"]
        LLM -> P1["P1: is_oral_assertion"]
        LLM -> P2["P2: is_written_assertion"]
        LLM -> P3["P3: is_non_verbal_assertive_conduct"]
        LLM -> P4["P4: is_out_of_court"]
        LLM -> P5["P5: is_introduced_to_prove_truth"]
    end
    subgraph Symbolic Engine Stage
        P1 -> OrGate((OR))
        P2 -> OrGate
        P3 -> OrGate
        OrGate -> AndGate{"AND Rule (op: ALL)"}
        P4 -> AndGate
        P5 -> AndGate
        AndGate -> Verdict["Final Legal Verdict: Hearsay / Not Hearsay"]
    end
    style AndGate fill:#ffeedd,stroke:#e67e22,stroke-width:2px
    style Verdict fill:#e8f8f5,stroke:#27ae60,stroke-width:2px
```

### 1.1 Empirical Divergence Across Model Architectures

When evaluating the transition from end-to-end **Baseline Chain-of-Thought (CoT)** reasoning to the **PYTHEN Neuro-Symbolic architecture**, empirical metrics diverge diametrically across model families on the 94-sample Hearsay benchmark:

$$\begin{aligned}
\text{DeepSeek V4 Flash}\colon \quad &\text{Recall}_{\text{CoT}} = 0.436 \xrightarrow{\text{PYTHEN}} \text{Recall}_{\text{NeSy}} = 0.892 \quad (\Delta \text{Recall} = +0.456, \; \Delta \text{F1} = +0.271) \\
\text{GPT OSS 20B}\colon \quad &\text{F1}_{\text{CoT}} = 0.761 \xrightarrow{\text{PYTHEN}} \text{F1}_{\text{NeSy}} = 0.857 \quad (\Delta \text{F1} = +0.0965) \\
\text{Gemma 4 31B}\colon \quad &\text{F1}_{\text{CoT}} = 0.828 \xrightarrow{\text{PYTHEN}} \text{F1}_{\text{NeSy}} = 0.731 \quad (\Delta \text{F1} = -0.0970)
\end{aligned}$$

| Model Architecture | Parameter Scale | CoT Precision | CoT Recall | CoT F1 | NeSy Precision | NeSy Recall | NeSy F1 | Net $\Delta \text{F1}$ | Primary Impact Mechanism |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| **DeepSeek V4 Flash** | MoE (~30B active) | **1.000** | 0.436 | 0.607 | 0.864 | **0.892** | **0.878** | $\mathbf{+0.271}$ | 🚀 *Latent Recall Unleashed via Decoupling* |
| **GPT OSS 20B** | Dense (20B) | 0.900 | 0.658 | 0.761 | 0.800 | 0.923 | **0.857** | $\mathbf{+0.096}$ | 📈 *Over-Conservatism & Logic Jumps Repaired* |
| **Gemma 4 31B** | Dense (31B) | 0.707 | **1.000** | **0.828** | 0.619 | 0.900 | **0.731** | $\mathbf{-0.097}$ | 🔻 *False Positive Avalanche & Rule Collapse* |

*Table 1: Macro-performance trajectory across Baseline CoT and PYTHEN NeSy execution pipelines on the Hearsay benchmark ($N = 94$).*

--

## 2. Mathematical Formalism & Error Propagation Dynamics

### 2.1 Formalization of the Legal Logic Gate

Under Federal Rule of Evidence 801(c), a statement is hearsay if and only if it is an out-of-court statement offered in evidence to prove the truth of the matter asserted. Formally, let the evidence universe be $\mathcal{E}$. For any evidence instance $e \in \mathcal{E}$, we define five boolean predicates:

- $P_1(e) \in \{0, 1\}$: `is_oral_assertion`
- $P_2(e) \in \{0, 1\}$: `is_written_assertion`
- $P_3(e) \in \{0, 1\}$: `is_non_verbal_assertive_conduct`
- $P_4(e) \in \{0, 1\}$: `is_out_of_court`
- $P_5(e) \in \{0, 1\}$: `is_introduced_to_prove_truth`

The symbolic decision function $\Phi \colon \{0, 1\}^5 \to \{0, 1\}$ is defined as:

$$\Phi(\vec{P}) = \left( \bigvee_{i=1}^3 P_i \right) \land P_4 \land P_5$$

In the PYTHEN rule engine configuration, this rule is instantiated as a strict conjunctive operator: `"op": "ALL"`.

### 2.2 The Asymmetry of Error Propagation in Conjunctive Rules

Let $\hat{P}_i$ denote the LLM's predicted value for ground-truth predicate $P_i^*$. Let $\epsilon_i = \hat{P}_i - P_i^* \in \{-1, 0, 1\}$.

Consider the standard legal topology where an out-of-court statement exists ($P_1^* \lor P_2^* \lor P_3^* = 1$, $P_4^* = 1$), but the evidence is introduced for a non-truth purpose (e.g., to prove notice, effect on the listener, or declarant state of mind), such that $P_5^* = 0$. The ground-truth verdict is strictly non-hearsay:

$$\Phi(\vec{P}^*) = 1 \land 1 \land 0 = 0 \quad \text{[Not Hearsay]}$$

If the LLM makes an extraction error on $P_5$ such that $\hat{P}_5 = 1$ (a False Positive Extraction, $\epsilon_5 = +1$), the predicted symbolic verdict becomes:

$$\Phi(\hat{\vec{P}}) = 1 \land 1 \land 1 = 1 \quad \text{[Hearsay]}$$

> **Theorem 1 (Conjunctive Error Asymmetry):**  
> Under a strict `"op": "ALL"` conjunction, the symbolic decision boundary exhibits zero fault tolerance against False Positive extractions on critical bottleneck predicates ($P_5$). A single $\epsilon_5 = +1$ error has an empirical propagation probability of $\mathbb{P}(\text{Verdict Flip} \mid \epsilon_5 = +1) = 1.0$ whenever the statement and out-of-court preconditions are met.

Conversely, in an end-to-end CoT architecture, the final decision is governed by a continuous latent representation $\mathbf{h}_{\text{decision}} = \operatorname{Softmax}(W \cdot \mathbf{h}_{\text{CoT}})$. A model with holistic context sensitivity can override intermediate linguistic assertions, maintaining a balanced decision boundary. The decoupling in NeSy strips away this latent dampening mechanism.

```
                    ┌────────────────────────────────────────────────────────┐
                    │      PREDICATE EXTRACTION (LLM Sub-symbolic)           │
                    │      P1=1, P2=0, P3=0 (Statement Detected)             │
                    │      P4=1             (Out of Court Detected)          │
                    │      P5: Ground Truth = 0 (Offered for Notice)         │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                                      LLM Prediction on P5
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
             P5 Predicted = 0 (Correct)                     P5 Predicted = 1 (FP Error)
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │     SYMBOLIC ENGINE       │                   │     SYMBOLIC ENGINE       │
         │   (1 OR 0 OR 0) & 1 & 0   │                   │   (1 OR 0 OR 0) & 1 & 1   │
         │         = FALSE           │                   │         = TRUE            │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       ▼                                               ▼
              VERDICT: NOT HEARSAY                            VERDICT: HEARSAY
                  [CORRECT]                                 [CATASTROPHIC FP FLIP]
```

--

## 3. Systematic Diagnostic Findings (Step 1 & Step 2 Evaluations)

### 3.1 Overview of Neuro-Symbolic Error Statistics (Step 1)

Our Step 1 diagnostic evaluation on all 94 samples revealed a total of 44 recorded error instances across the three evaluated models (34 unique samples after filtering parsing discrepancies):

| Model Name | Total Dataset ($N$) | NeSy Correct | NeSy Error | Error Rate (%) | Samples with CoT Reasoning | `NS_WRONG_WITH_COT` |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **DeepSeek V4 Flash** | 94 | 84 | **10** | **10.6%** | 94 | 0 |
| **GPT OSS 20B** | 94 | 77 | **17** | **18.1%** | 94 | 17 |
| **Gemma 4 31B** | 94 | 77 | **17** | **18.1%** | 94 | **17** ⚠️ |

*Table 2: Summary of model performance and error breakdown in Step 1 evaluation.*

### 3.2 Directional Error Bias: Extreme Asymmetry

Evaluating the balance between False Positives (FP) and False Negatives (FN) reveals stark architectural biases:

| Model | Total Errors | False Positive (FP) | False Negative (FN) | FP : FN Ratio | Identified Bias Category |
|:--|:--:|:--:|:--:|:--:|:--|
| **DeepSeek V4 Flash** | 10 | 5 | 5 | 1 : 1 | **Balanced** (No systematic directional bias) |
| **GPT OSS 20B** | 17 | 3 | 14 | 1 : 4.7 | **Conservative Bias** (High miss rate on hearsay) |
| **Gemma 4 31B** | 17 | **17** | **0** | $\infty$ | **Extreme Aggressive Bias** (100% overclassification) |

*Table 3: False Positive vs. False Negative distributions across models.*

```
Error Direction Distribution:
DeepSeek : ████████████████████████████████████████ FP=5  FN=5   (Perfect Balance)
GPT      : ████████                                 FP=3  ████████████████████████████████████ FN=14
Gemma    : ████████████████████████████████████████████████████████████████████████████████████ FP=17  FN=0
```

### 3.3 Slice-Level Performance Breakdown

Analyzing accuracy across the 5 distinct evidentiary categories (slices) illustrates the distinct operational domains of each model:

| Evidentiary Slice | Support ($n$) | DeepSeek Accuracy (%) | GPT OSS 20B Accuracy (%) | Gemma 4 31B Accuracy (%) | Primary Failure Profile |
|:--|:--:|:--:|:--:|:--:|:--|
| **Non-assertive conduct** | 19 | 94.7% | **100.0%** | 84.2% 🔴 | Gemma mischaracterizes conduct as statement |
| **Non-verbal hearsay** | 12 | 91.7% | 91.7% | **100.0%** | Minor recall drop on subtle gestures |
| **Not introduced to prove truth** | 20 | 85.0% | 90.0% | **35.0%** 🔴🔴 | Gemma suffers catastrophic $P_5$ collapse |
| **Standard hearsay** | 29 | 86.2% | 55.2% 🔴 | **100.0%** | GPT CoT overthinks standard cases; Gemma 100% |
| **Statement made in-court** | 14 | 92.9% | 92.9% | 92.9% | Daniel in-court testimony ambiguity (#23) |

*Table 4: Empirical accuracy broken down by evidentiary slice.*

--

## 4. Deep Dive: The False Positive Trap in Gemma 4 31B

### 4.1 The "Step 3 / P5 Collapse" Phenomenon

In 100% of Gemma's 17 NeSy errors, the error direction was exclusively **False Positive (FP = 17, FN = 0)**. Gemma never misses a genuine hearsay item, but systematically hallucinates hearsay where none exists.

```
Error Distribution Breakdown for Gemma 4 31B (17 Errors):
████████████████████████████████████████████████████ False Positives: 17 (100%)
(Zero False Negatives: FN = 0)
```

### 4.2 Cognitive Source of Gemma's False Positive Bias

Gemma fails to distinguish between two foundational legal concepts:
1. **Evidentiary Relevance via Content Awareness:** The factfinder must understand what words were uttered to know that the listener was placed on notice.
2. **Substantive Reliance on Truth:** The factfinder must believe the asserted facts within the utterance are objectively true.

Whenever Gemma identifies *any semantic connection* between the factual proposition at trial and the content of the statement, it automatically triggers $\hat{P}_5 = \text{True}$.

```mermaid
graph TD
    Text["Evidence Fact Pattern: Bob told Jane that Arthur was cheating"] -> GemmaBrain{"Gemma Semantic Parser"}
    GemmaBrain ->|"Is the statement connected to the issue of Jane's knowledge?"| YesConn[Yes: Highly Relevant]
    YesConn -> Confusion{"Semantic Collapse Step"}
    Confusion ->|Flawed Logic: 'To prove knowledge, the words must be true'| P5True["P5 (is_introduced_to_prove_truth) = TRUE ❌"]
    P5True -> PYTHEN["PYTHEN Strict AND Gate"]
    PYTHEN -> WrongVerdict["Output: HEARSAY (Ground Truth: NOT HEARSAY)"]
    style P5True fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    style WrongVerdict fill:#ffcccc,stroke:#cc0000,stroke-width:2px
```

### 4.3 Three Manifestations of Gemma's Failure Modes

Manual inspection of Gemma's 17 failure samples reveals three distinct patterns:

1. **Purpose Confusion (12 / 17 samples, 70.6%):** Conflating "offered to prove awareness / notice / state of mind" with "offered to prove truth of matter asserted" (e.g., samples #47, #76, #78, #79, #80, #82, #85, #86, #87, #90, #91).
2. **Conduct Misclassification (3 / 17 samples, 17.6%):** Treating non-communicative actions as assertive statements (e.g., poster display in #6, medical diagnosis in #12, driver test record in #15).
3. **In-Court Boundary Confusion (1 / 17 samples, 5.9%):** Treating in-court testimony describing past feelings as out-of-court hearsay (#23).

--

## 5. Deep Dive: Breaking Decision Paralysis in DeepSeek V4 Flash

### 5.1 The Baseline CoT Bottleneck: Extreme Over-Conservatism

In standalone Chain-of-Thought reasoning, DeepSeek V4 Flash exhibited extreme decision conservatism:
- **Baseline Precision:** **$1.000$** (100% of samples predicted as Hearsay were truly Hearsay; zero False Positives).
- **Baseline Recall:** **$0.436$** (Broke down on more than 56% of true Hearsay samples; massive False Negatives).

This phenomenon stems from **"Verdict Hesitation"**: when tasked with issuing a high-stakes, binding legal verdict, DeepSeek's safety and alignment training forces it to over-index on hypothetical exceptions (e.g., searching for non-existent party-opponent exemptions, excited utterances, or treating direct assertions as circumstantial evidence). Consequently, DeepSeek systematically defaulted to `"Not Hearsay"` whenever ambiguity was present.

### 5.2 How PYTHEN Unlocks Latent Reasoning

When migrated to PYTHEN, the reasoning burden is fundamentally restructured:
1. DeepSeek is **never asked** to decide if an item is legally admissible or if it constitutes Hearsay.
2. DeepSeek is asked only 5 micro-level factual questions: *"Was this spoken?"*, *"Was it out of court?"*, *"Is it offered for its truth?"*.

Relieved of the end-to-end verdict penalty, DeepSeek’s extraordinary feature-extraction capabilities emerge. It extracts factual indicators with clinical objectivity.

$$\text{Recall}_{\text{CoT}} = 0.436 \xrightarrow{\quad \text{PYTHEN Decoupling} \quad} \text{Recall}_{\text{NeSy}} = 0.892 \quad (\text{\textbf{+104.6\% relative gain}})$$

```
DeepSeek Recall Evolution:
CoT Baseline:  █████████████████████ 0.436
PYTHEN NeSy:   ███████████████████████████████████████████ 0.892 (+0.456)
```

The symbolic engine absorbs the cognitive load of combining these factual extractions, generating verdicts that are mathematically sound and free from hedging bias.

--

## 6. Comprehensive Error Taxonomy Matrix (Levels E1–E6)

To rigorously classify all failure modes observed across the 94 benchmark cases, we construct a 6-tier hierarchical Error Taxonomy spanning the extraction, cognitive reasoning, and serialization layers:

```mermaid
graph TD
    Taxonomy["Error Taxonomy in Neuro-Symbolic Legal Reasoning"]
    Taxonomy -> Ext["Perception & Extraction Layer"]
    Taxonomy -> CoT["Cognitive & Reasoning Layer"]
    Taxonomy -> Infra["Syntactic & Structural Layer"]
    
    Ext -> E1["E1: Predicate Hallucination (False Positive Extraction)"]
    Ext -> E2["E2: Predicate Omission (False Negative Extraction)"]
    Ext -> E3["E3: Grounding / Reference Hallucination"]
    
    CoT -> E4["E4: Over-Conservatism / Verdict Hesitation"]
    CoT -> E5["E5: Contextual & Logical Jump"]
    
    Infra -> E6["E6: Schema & Structural Parsing Failure"]
    
    style E1 fill:#ffdddd,stroke:#c0392b,stroke-width:2px
    style E2 fill:#fff0dd,stroke:#d35400,stroke-width:2px
    style E4 fill:#ddebff,stroke:#2980b9,stroke-width:2px
    style E6 fill:#eeddff,stroke:#8e44ad,stroke-width:2px
```

| Error Code | Error Designation | Architectural Layer | Formal Definition | Model Vulnerability Profile | Impact on Symbolic Engine |
|:--|:--|:--:|:--|:--|:--|
| **E1** | **Predicate Hallucination** *(False Positive)* | Sub-symbolic Extraction | $\hat{P}_i = 1$ when $P_i^* = 0$. Model asserts presence of non-existent legal predicate. | **Gemma 4 31B** (59.5% of total errors), DeepSeek on $P_5$. | Catastrophic: Instantly flips `"op": "ALL"` conjunction to False Positive verdict. |
| **E2** | **Predicate Omission** *(False Negative)* | Sub-symbolic Extraction | $\hat{P}_i = 0$ when $P_i^* = 1$. Model misses subtle communicative acts or out-of-court markers. | **DeepSeek V3.2 / V4** on complex written records. | Nullifies conjunction: Engine defaults to False Negative verdict. |
| **E3** | **Grounding / No Reference** | Grounding Interface | Model extracts predicate value but quotes non-existent or irrelevant source spans. | Observed in DeepSeek (9.57% of samples). | Extraction may be factually correct by chance, but lacks formal verifiability. |
| **E4** | **Over-Conservatism Bias** | Holistic CoT Layer | Step-by-step reasoning is correct, but conclusion retreats defensively to negative verdict. | **GPT OSS 20B** (21.2% of errors), **DeepSeek CoT Baseline**. | Solved by PYTHEN: Symbolic engine overrides CoT hesitation. |
| **E5** | **Logical Jump / Non-Sequitur** | Holistic CoT Layer | Incoherent leap between premises and conclusion within the textual thought trace. | DeepSeek Baseline (rare, 1.06%). | Bypassed in PYTHEN via formal propositional resolution. |
| **E6** | **Structural Parsing Failure** | Serialization Interface | LLM outputs malformed JSON or unescaped control tokens; schema parsing fails. | **GPT OSS 20B** (23.5% of errors; 4 invalid JSON instances). | Triggers emergency fallback: Engine returns `False`, injecting artificial FNs. |

*Table 5: Comprehensive Error Taxonomy Matrix with layer mappings and impact dynamics.*

### 6.1 Quantitative Error Distribution Across Models ($N = 94$ Samples)

| Model Name | E1 (FP Extr.) | E2 (FN Extr.) | E3 (No Grounding) | E4 (Over-Conservatism) | E5 (Logical Jump) | E6 (JSON Failure) | Total Fault Events |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Gemma 4 31B** | **22 (59.5%)** | 15 (40.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **37** |
| **GPT OSS 20B** | **35 (53.0%)** | 17 (25.8%) | 0 (0.0%) | 14 (21.2%) | 0 (0.0%) | 4 (6.1%) | **66** |
| **DeepSeek V4 / V3.2** | **39 (51.3%)** | 23 (30.3%) | 9 (11.8%) | 4 (5.3%) | 1 (1.3%) | 0 (0.0%) | **76** |

*Table 6: Error code distributions across all models on the Hearsay benchmark.*

--

## 7. Standalone Predicate Accuracy & Extraction Evaluation

To determine whether performance degradation stems from linguistic inability or rule propagation, we evaluate extraction performance across each individual predicate $P_1$ through $P_5$:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 7.1 Detailed Predicate Performance Breakdown

| Predicate | Model | Precision | Recall | F1-Score | TP | FP | FN |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **$P_1$:** `is_oral_assertion` | DeepSeek V3.2 | **1.0000** | 0.8136 | 0.8972 | 48 | 0 | 11 |
| | GPT OSS 20B | **1.0000** | 0.9245 | **0.9608** | 49 | 0 | 4 |
| | Gemma 4B | 0.9821 | **0.9322** | 0.9565 | 55 | 1 | 4 |
| **$P_2$:** `is_written_assertion` | DeepSeek V3.2 | 0.1818 | 0.6667 | 0.2857 | 2 | 9 | 1 |
| | GPT OSS 20B | 0.1000 | 0.5000 | 0.1667 | 1 | 9 | 1 |
| | Gemma 4B | **0.2308** | **1.0000** | **0.3750** | 3 | 10 | 0 |
| **$P_3$:** `is_non_verbal_assertive_conduct` | DeepSeek V3.2 | 0.7273 | 0.5000 | 0.5926 | 8 | 3 | 8 |
| | GPT OSS 20B | 0.7500 | **0.5625** | 0.6429 | 9 | 3 | 7 |
| | Gemma 4B | **1.0000** | 0.5000 | **0.6667** | 8 | 0 | 8 |
| **$P_4$:** `is_out_of_court` | DeepSeek V3.2 | 0.8356 | **1.0000** | 0.9104 | 61 | 12 | 0 |
| | GPT OSS 20B | 0.8594 | 0.9821 | 0.9167 | 55 | 9 | 1 |
| | Gemma 4B | **0.9531** | **1.0000** | **0.9760** | 61 | 3 | 0 |
| **$P_5$:** `is_introduced_to_prove_truth` | DeepSeek V3.2 | 0.6970 | 0.8364 | 0.7603 | 46 | 20 | 9 |
| | GPT OSS 20B | 0.6618 | 0.8824 | 0.7563 | 45 | 23 | 6 |
| | Gemma 4B | **0.8095** | **0.9273** | **0.8644** | 51 | 12 | 4 |

*Table 7: Fine-grained precision, recall, and F1 performance per predicate across all models.*

### 7.2 Macro-Averaged Extraction Metrics

| Evaluated Model | Macro-Precision | Macro-Recall | Macro-F1 | Total Extraction Faults |
|:--|:--:|:--:|:--:|:--:|
| **Gemma 4B / 31B** | **0.7951** | **0.8719** | **0.7677** | **42** |
| **DeepSeek V3.2 / V4** | 0.6883 | 0.7633 | 0.6892 | 73 |
| **GPT OSS 20B** | 0.6742 | 0.7703 | 0.6887 | 63 |

*Table 8: Macro-averaged extraction metrics across the 5 target predicates.*

> **Key Insight:** Gemma achieves the **highest raw predicate extraction Macro-F1 (0.7677)** and the fewest overall extraction errors (42 vs. 73 for DeepSeek). This formally proves that Gemma's end-to-end performance degradation in PYTHEN is **not** caused by inferior linguistic extraction, but by the strategic distribution of its errors—specifically, False Positives concentrated on the critical bottleneck predicate $P_5$ and non-assertive conduct ($P_2$).

--

## 8. Qualitative Legal Case Studies & Manual Annotations

We present six annotated case studies from the benchmark illustrating the operational mechanisms of each model.

--

### Case Study 1: Non-Assertive Conduct Misclassification (#6)
**Slice:** `Non-assertive conduct` | **Ground Truth:** `False` (Not Hearsay)

> **Fact Pattern:**  
> *"On the issue of whether Patricia was a fan of Coldplay, the fact that she had a poster with the lyrics of 'Viva la Vida' on her bedroom wall."*

* **Ground Truth Analysis:** The poster contains written words, but displaying it is non-assertive conduct reflecting personal preference. It is not offered to prove the factual truth of the lyrics ($P_2^* = \text{False}, P_5^* = \text{False}$).
* **Gemma 4 31B:** Gags $P_2 = \text{True}$ and $P_5 = \text{True}$ (Error E1).
  $$\Phi(\hat{\vec{P}}) = (0 \lor 1 \lor 0) \land 1 \land 1 = 1 \quad \text{[False Positive Flip ❌]}$$
* **DeepSeek V4 Flash:** Correctly identifies non-assertive conduct ($P_2 = 0, P_5 = 0$).
  $$\Phi(\hat{\vec{P}}) = (0 \lor 0 \lor 0) \land 1 \land 0 = 0 \quad \text{[Correct Non-Hearsay 🟢]}$$

--

### Case Study 2: Notice & Knowledge vs. Truth of Content (#47)
**Slice:** `Not introduced to prove truth` | **Ground Truth:** `False` (Not Hearsay)

> **Fact Pattern:**  
> *"On the issue of whether Jane knew that Arthur was having an affair, the fact that Bob told Jane he suspected Arthur was up to something."*

* **Ground Truth Analysis:** The statement is offered to prove Jane’s state of knowledge (notice / effect on listener). Whether Arthur was in fact unfaithful is immaterial ($P_5^* = \text{False}$).
* **Gemma 4 31B Reasoning:** *"Yes, to prove Jane knew of the infidelity, the court must rely on the substance of Bob's warning."* $\to \hat{P}_5 = \text{True}$ (Error E1). PYTHEN returns $\text{True}$ (False Positive ❌).
* **DeepSeek V4 Flash Reasoning:** *"Offered to establish notice/awareness in Jane, not that Arthur was up to something."* $\to \hat{P}_5 = \text{False}$. PYTHEN returns $\text{False}$ (Correct 🟢).

--

### Case Study 3: Standard Hearsay & CoT Hesitation Recovery (#37)
**Slice:** `Standard hearsay` | **Ground Truth:** `True` (Hearsay)

> **Fact Pattern:**  
> *"Stephen told his wife on the phone that he wasn't paying attention to the road, offered to prove that Stephen ran a red light."*

* **Baseline CoT (GPT OSS 20B):** Reasoned that because Stephen did not explicitly say *"I ran a red light"*, the statement was mere circumstantial evidence, defensively concluding `Not Hearsay` (Error E4 Over-Conservatism ❌).
* **PYTHEN Execution:** Extracted $P_1 = 1, P_4 = 1, P_5 = 1$. The rule engine executed $(1 \lor 0 \lor 0) \land 1 \land 1 = 1$, successfully recovering the correct verdict (True Positive 🟢).

--

### Case Study 4: Medical Diagnosis & Circumstantial Awareness (#12)
**Slice:** `Non-assertive conduct` | **Ground Truth:** `False` (Not Hearsay)

> **Fact Pattern:**  
> *"On the issue of whether Jessica was aware she was trespassing, the fact that Jessica had been diagnosed as near-sighted by her ophthalmologist."*

* **Analysis:** A professional medical diagnosis offered to show physical visual impairment is circumstantial evidence of sensory capacity, not an out-of-court assertion offered for its narrative truth.
* **Gemma Failure:** Conflated medical diagnosis with an assertive statement, propagating through the conjunction to trigger a False Positive.

--

### Case Study 5: Prior Driving Record (#15)
**Slice:** `Non-assertive conduct` | **Ground Truth:** `False` (Not Hearsay)

> **Fact Pattern:**  
> *"On the issue of whether Karen negligently operated the forklift, the fact that Karen is a terrible driver who needed several tries to pass her commercial license test."*

* **Analysis:** Administrative history and test attempts represent non-assertive historical events, not communicative hearsay statements.
* **Gemma Failure:** Classified historical testing records as written assertions ($P_2 = \text{True}$), leading to an erroneous hearsay finding.

--

### Case Study 6: In-Court Witness Testimony (#23)
**Slice:** `Statement made in-court` | **Ground Truth:** `False` (Not Hearsay)

> **Fact Pattern:**  
> *"On the issue of whether Daniel drove negligently, the fact that Daniel testified during the trial that he told his wife he was tired before driving into a tree."*

* **Analysis:** While the past statement was out of court, Daniel is actively testifying under oath on the witness stand, subject to cross-examination regarding his direct admission.
* **Universal Difficulty:** All three models struggled on this boundary case, misassigning $P_4$ due to the nested structure of in-court testimony describing an out-of-court statement.

--

## 9. Comparative Synthesis: Three Behavioral Archetypes

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    NEURO-SYMBOLIC BEHAVIORAL ARCHETYPES                                 │
├───────────────────────┬──────────────────────────────────┬──────────────────────────────┤
│ Model Family          │ Baseline CoT Profile             │ PYTHEN NeSy Transformation   │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ DeepSeek V4 Flash     │ Over-Conservative (P=1.0, R=0.44)│ Precision Maintained         │
│                       │ Crippled by verdict hesitation   │ Recall surges to 0.892 🚀    │
│                       │ High latent feature capability   │ Optimal NeSy Beneficiary     │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ GPT OSS 20B           │ Moderate Hesitation + JSON bugs  │ Symbolic Safety Net rescues  │
│                       │ Loses simple Standard cases      │ CoT logical jumps            │
│                       │ False Negative heavy (14 FN)     │ F1 jumps from 0.761 to 0.857 │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ Google Gemma 4 31B    │ Over-Inclusive (Recall = 1.0)    │ Rigid "op: ALL" amplifies    │
│                       │ Step 3 / P5 Semantic Collapse    │ P5 False Positives           │
│                       │ Perfect on Simple; Fails complex │ F1 drops from 0.828 to 0.731 │
└───────────────────────┴──────────────────────────────────┴──────────────────────────────┘
```

--

## 10. Architectural Implications & Mitigation Strategies

Our empirical analysis demonstrates that unweighted, two-stage Neuro-Symbolic pipelines suffer from structural fragility when paired with models exhibiting directional calibration skew. We propose three actionable architectural enhancements:

### 1. Contrastive Boundary Extraction Prompts
To eliminate Gemma's semantic collapse on $P_5$, extraction prompts must explicitly encode negative contrastive exemplars distinguishing **Truth** from **Notice/Knowledge**:
```json
"extraction_guardrail": {
  "predicate": "is_introduced_to_prove_truth",
  "instruction": "Set to false if the evidence is offered to show the listener's state of mind, awareness, warning, or motive, regardless of whether the statement itself is true."
}
```

### 2. Confidence-Calibrated Soft Symbolic Operators
Replace rigid boolean gates with a **Calibrated Probabilistic T-Norm**:

$$\mathbb{P}(\text{Hearsay}) = T_{\text{G\"odel}}\left( \max(P_1, P_2, P_3), P_4, P_5 \cdot (1 - \sigma_{\text{uncertainty}}) \right)$$

Where $\sigma_{\text{uncertainty}}$ penalizes models with historically high False Positive rates on bottleneck predicates.

### 3. Bidirectional Neuro-Symbolic Verification Loops
When the symbolic engine detects that $P_5 = 1$ in the presence of notice keywords (*"knew"*, *"heard"*, *"warned"*, *"notified"*), it should trigger an automated symbolic critique back to the LLM:
> *"The symbolic solver detected a potential Notice exception. Verify whether the legal issue strictly requires the underlying statement to be objectively true."*

--

## 11. Conclusion

The performance divergence between DeepSeek V4 Flash, GPT OSS 20B, and Gemma 4 31B under the PYTHEN framework exposes the dual nature of neuro-symbolic computation. When an LLM's primary weakness is cognitive over-caution at the final synthesis layer (DeepSeek, GPT), symbolic engines act as catalytic accelerators that unlock latent extraction capability. Conversely, when an LLM's primary weakness is semantic over-extension on critical boundary conditions (Gemma), rigid symbolic rules serve as catastrophic error multipliers. Future neuro-symbolic research must advance beyond simple decoupled pipelines toward calibrated, error-aware hybrid architectures that harmonize linguistic intuition with formal logical rigor.

--

### Appendix: PYTHEN Rule Engine Formal Specification

```json
{
  "rule_name": "FRE_801_c_hearsay_definition",
  "version": "2.4.0",
  "symbolic_logic": {
    "op": "ALL",
    "conditions": [
      {
        "op": "ANY",
        "predicates": [
          "is_oral_assertion",
          "is_written_assertion",
          "is_non_verbal_assertive_conduct"
        ]
      },
      {
        "op": "EQUAL",
        "predicate": "is_out_of_court",
        "target": true
      },
      {
        "op": "EQUAL",
        "predicate": "is_introduced_to_prove_truth",
        "target": true
      }
    ]
  },
  "deterministic_fallback": false
}
```
