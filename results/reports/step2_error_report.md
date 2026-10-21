# Step 2 — Phân Tích Lỗi Sâu: Error Taxonomy, Manual Annotation & Predicate Accuracy
**Bộ dữ liệu:** Hearsay (94 mẫu) | **Ngày thực hiện:** 21/08/2026

--

## 1. Tổng Quan

| Mô hình | File PYTHEN | File Baseline |
|--|--|--|
| Gemma 4B | `pythen_google.gemma-4b.csv` | `google.gemma-4-31b.csv` |
| DeepSeek V3.2 | `pythen_deepseek-v3.2.csv` | `deepseek.deepseek-v4-flash.csv` |
| GPT OSS 20B | `pythen_openai.gpt-oss-20b.csv` | `openai.gpt-5.6-luna.csv` |

--

## 2. Error Taxonomy — Tầng LLM Extraction (E1/E2/E3)

### 2.1 Định nghĩa

| Mã | Tên | Mô tả |
|--|--|--|
| **E1** | Predicate Hallucination (FP) | LLM gán `true` khi GT=False |
| **E2** | Predicate Omission (FN) | LLM gán `false` khi GT=True |
| **E3** | Quote Mismatch / No Reference | LLM=True, GT=True nhưng reference_text=None |

### 2.2 Phân Phối Lỗi (34 mẫu sai từ Step 1)

| Model | Tổng sai | E1 | E2 | E3 | No Pred Error |
|--|:--:|:--:|:--:|:--:|:--:|
| Gemma 4B | 17 | 11 | 0 | 0 | 6 |
| GPT OSS 20B | 15 | 3 | 3 | 0 | 10 |

### 2.3 Predicate Hay Bị Lỗi Nhất (Gemma 4B — 17 mẫu sai)

| Critical Predicate | Số mẫu |
|--|:--:|
| `is_introduced_to_prove_truth` | 9 |
| `` | 6 |
| `is_written_assertion` | 1 |
| `is_out_of_court` | 1 |

--

## 3. Error Taxonomy — Tầng Baseline CoT (E4/E5)

### 3.1 Định Nghĩa

| Mã | Tên | Nhận diện |
|--|--|--|
| **E4** | Over-conservatism Bias | step3 đúng nhưng conclusion phủ nhận |
| **E5** | Logical Jump in CoT | step3 bỏ qua điều kiện pháp lý |

### 3.2 Phân Phối Lỗi Baseline CoT

| Model | Tổng sai | E4 | E5 | Other |
|--|:--:|:--:|:--:|:--:|
| Gemma 4B | 17 | 0 | 0 | 17 |
| DeepSeek V3.2 | 10 | 4 | 1 | 5 |
| GPT OSS 20B | 17 | 14 | 0 | 3 |

--

## 4. Manual Annotation — 5 Mẫu Đại Diện (Gemma 4B)

**Sample #6** | Slice: `Non-assertive conduct`
- Fact Pattern: _On the issue of whether Patricia was a fan of Coldplay, the fact that she had a poster with the lyrics of "Viva la Vida" on her bedroom wall...._
- GT Hearsay: `False` | Gemma Prediction: `True`
- **Lỗi:** `E1`
- **Chi tiết:** E1: is_written_assertion LLM=True GT=False | E1: is_out_of_court LLM=True GT=False | E1: is_introduced_to_prove_truth LLM=True GT=False
- **Hệ quả PYTHEN:** GT=False, PYTHEN=True → False Positive (phan loai nham)

**Sample #12** | Slice: `Non-assertive conduct`
- Fact Pattern: _On the issue of whether Jessica was aware she was trespassing, the fact that Jessica had been diagnosed as near-sighted by her ophthalmologist...._
- GT Hearsay: `False` | Gemma Prediction: `True`
- **Lỗi:** `NO_PREDICATE_ERROR`
- **Chi tiết:** No predicate-level error
- **Hệ quả PYTHEN:** GT=False, PYTHEN=True → False Positive (phan loai nham)

**Sample #15** | Slice: `Non-assertive conduct`
- Fact Pattern: _On the issue of whether Karen negligently operated the forklift, the fact that Karen is a terrible driver who needed several tries to pass her commercial license test...._
- GT Hearsay: `False` | Gemma Prediction: `True`
- **Lỗi:** `NO_PREDICATE_ERROR`
- **Chi tiết:** No predicate-level error
- **Hệ quả PYTHEN:** GT=False, PYTHEN=True → False Positive (phan loai nham)

**Sample #23** | Slice: `Statement made in-court`
- Fact Pattern: _On the issue of whether Daniel drove negligently, the fact that Daniel testified during the trial that he told his wife he was tired before driving into a tree...._
- GT Hearsay: `False` | Gemma Prediction: `True`
- **Lỗi:** `E1`
- **Chi tiết:** E1: is_out_of_court LLM=True GT=False
- **Hệ quả PYTHEN:** GT=False, PYTHEN=True → False Positive (phan loai nham)

**Sample #47** | Slice: `Not introduced to prove truth`
- Fact Pattern: _On the issue of whether Jane knew that Arthur was having an affair, the fact that Bob told Jane he suspected Arthur was up to something...._
- GT Hearsay: `False` | Gemma Prediction: `True`
- **Lỗi:** `NO_PREDICATE_ERROR`
- **Chi tiết:** No predicate-level error
- **Hệ quả PYTHEN:** GT=False, PYTHEN=True → False Positive (phan loai nham)

--

## 5. Predicate Accuracy (94 mẫu)

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

### 5.1 Chi Tiết theo Model x Predicate

| Predicate | Model | Prec | Recall | F1 | TP | FP | FN |
|--|--|--:|--:|--:|--:|--:|--:|
| `is_introduced_to_prove_truth` | DeepSeek V3.2 | 0.6970 | 0.8364 | 0.7603 | 46 | 20 | 9 |
| `is_introduced_to_prove_truth` | GPT OSS 20B | 0.6618 | 0.8824 | 0.7563 | 45 | 23 | 6 |
| `is_introduced_to_prove_truth` | Gemma 4B | 0.8095 | 0.9273 | 0.8644 | 51 | 12 | 4 |
| `is_non_verbal_assertive_conduct` | DeepSeek V3.2 | 0.7273 | 0.5000 | 0.5926 | 8 | 3 | 8 |
| `is_non_verbal_assertive_conduct` | GPT OSS 20B | 0.7500 | 0.5625 | 0.6429 | 9 | 3 | 7 |
| `is_non_verbal_assertive_conduct` | Gemma 4B | 1.0000 | 0.5000 | 0.6667 | 8 | 0 | 8 |
| `is_oral_assertion` | DeepSeek V3.2 | 1.0000 | 0.8136 | 0.8972 | 48 | 0 | 11 |
| `is_oral_assertion` | GPT OSS 20B | 1.0000 | 0.9245 | 0.9608 | 49 | 0 | 4 |
| `is_oral_assertion` | Gemma 4B | 0.9821 | 0.9322 | 0.9565 | 55 | 1 | 4 |
| `is_out_of_court` | DeepSeek V3.2 | 0.8356 | 1.0000 | 0.9104 | 61 | 12 | 0 |
| `is_out_of_court` | GPT OSS 20B | 0.8594 | 0.9821 | 0.9167 | 55 | 9 | 1 |
| `is_out_of_court` | Gemma 4B | 0.9531 | 1.0000 | 0.9760 | 61 | 3 | 0 |
| `is_written_assertion` | DeepSeek V3.2 | 0.1818 | 0.6667 | 0.2857 | 2 | 9 | 1 |
| `is_written_assertion` | GPT OSS 20B | 0.1000 | 0.5000 | 0.1667 | 1 | 9 | 1 |
| `is_written_assertion` | Gemma 4B | 0.2308 | 1.0000 | 0.3750 | 3 | 10 | 0 |

### 5.2 Macro-Average per Model

| Model | Avg Precision | Avg Recall | Avg F1 |
|--|--:|--:|--:|
| DeepSeek V3.2 | 0.6883 | 0.7633 | 0.6892 |
| GPT OSS 20B | 0.6742 | 0.7703 | 0.6887 |
| Gemma 4B | 0.7951 | 0.8719 | 0.7677 |

### 5.3 Average F1 per Predicate (3 models)

| Predicate | Avg Precision | Avg Recall | Avg F1 |
|--|--:|--:|--:|
| `is_written_assertion` | 0.1709 | 0.7222 | 0.2758 |
| `is_non_verbal_assertive_conduct` | 0.8258 | 0.5208 | 0.6341 |
| `is_introduced_to_prove_truth` | 0.7228 | 0.8820 | 0.7937 |
| `is_out_of_court` | 0.8827 | 0.9940 | 0.9344 |
| `is_oral_assertion` | 0.9940 | 0.8901 | 0.9382 |

--

## 6. Phân Tích Tổng Hợp

> [!IMPORTANT]
> **Kết luận chính:** `is_introduced_to_prove_truth` là predicate dễ sai nhất — phản ánh nhầm lẫn giữa "offered to prove truth of content" vs "offered to prove knowledge/effect on listener".

> [!NOTE]
> **E3 (No Reference):** Predicate đúng nhưng thiếu trích dẫn — không ảnh hưởng PYTHEN logic nhưng giảm explainability.

> [!WARNING]
> **E4 dominant trong CoT:** Over-conservatism bias là lỗi hệ thống — model biết đúng nhưng không dám kết luận.

### So sánh 2 tầng

| Tầng | Lỗi chủ đạo | Nguyên nhân | Hệ quả |
|--|--|--|--|
| LLM Extraction (PYTHEN) | E1 trên `is_introduced_to_prove_truth` | Nhầm lẫn ngữ nghĩa pháp lý | PYTHEN ALL-operator → False |
| Baseline CoT | E4 Over-conservatism | Calibration bias | CoT reasoning đúng → result=False |

--

## 7. Output Files

| File | Mô tả | Số dòng |
|--|--|:--:|
| `step2_error_taxonomy_llm.csv` | E1/E2/E3 — LLM tier | 274 |
| `step2_error_taxonomy_cot.csv` | E4/E5 — CoT tier | 44 |
| `step2_annotation_sheet.csv` | Annotation log 34 mẫu sai | 32 |
| `step2_predicate_accuracy.csv` | Precision/Recall/F1 | 15 |

--
*Generated by Step 2 — PYTHEN Paper Experiments*