# Báo Cáo Đánh Giá Độc Lập Khâu Trích Xuất Vị Tự (Standalone Predicate Extraction Evaluation)

## Ablation Study – Bước 3: Đánh Giá Năng Lực Trích Xuất Ngôn Ngữ Thuần Túy

> **Mục tiêu**: Đánh giá năng lực trích xuất vị tự (predicate extraction) thuần túy của từng LLM, tách biệt khỏi pipeline suy luận PYTHEN, nhằm xác định nguồn gốc sai sót hiệu năng.

--

## 1. Phương Pháp Đánh Giá

### 1.1 Dữ liệu

| Thành phần | Mô tả | Số mẫu |
|--|--|--|
| **Ground Truth** | `hearsay_predicates_breakdown.csv` — 5 vị tự nhị phân cho 94 tình huống pháp lý | 94 |
| **DeepSeek-V3.2** | `pythen_deepseek-v3.2.csv` — trích xuất từ cột `llm_output` | 94 |
| **GPT-OSS-20B** | `pythen_openai.gpt-oss-20b.csv` — trích xuất từ cột `llm_output` | 86* |
| **Gemma-4B** | `pythen_google.gemma-4b.csv` — trích xuất từ cột `llm_output` | 94 |

> [!NOTE]
> *GPT-OSS-20B chỉ có 86 mẫu khớp với ground truth (thiếu 8 mẫu do parsing hoặc timeout), các chỉ số được tính trên tập mẫu chung.

### 1.2 Các Vị Tự Được Đánh Giá

| Ký hiệu | Vị tự | Mô tả |
|--|--|--|
| P₁ | `is_oral_assertion` | Lời nói/phát ngôn bằng miệng |
| P₂ | `is_written_assertion` | Khẳng định bằng văn bản |
| P₃ | `is_non_verbal_assertive_conduct` | Hành vi phi ngôn ngữ mang tính khẳng định |
| P₄ | `is_out_of_court` | Được thực hiện ngoài tòa án |
| P₅ | `is_introduced_to_prove_truth` | Được đưa ra để chứng minh sự thật |

### 1.3 Chỉ Số Đo Lường

Cho từng vị tự $p_i$:

$$\text{Precision}_{p_i} = \frac{TP_{p_i}}{TP_{p_i} + FP_{p_i}}$$

$$\text{Recall}_{p_i} = \frac{TP_{p_i}}{TP_{p_i} + FN_{p_i}}$$

$$\text{F1}_{p_i} = 2 \times \frac{\text{Precision}_{p_i} \times \text{Recall}_{p_i}}{\text{Precision}_{p_i} + \text{Recall}_{p_i}}$$

Tổng thể:
- **Macro-F1** = trung bình cộng F1 của 5 vị tự
- **Micro-F1** = F1 tính trên tổng TP/FP/FN gộp từ tất cả vị tự

--

## 2. Kết Quả Tổng Thể

### 2.1 So Sánh Chỉ Số Tổng Hợp

| Mô hình | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Accuracy | Samples |
|--|:--:|:--:|:--:|:--:|:--:|:--:|
| **Gemma-4B** | **0.7677** | **0.8945** | **0.8725** | **0.9175** | **0.9106** | 94 |
| GPT-OSS-20B | 0.6887 | 0.8346 | 0.7833 | 0.8933 | 0.8535 | 86 |
| DeepSeek-V3.2 | 0.6892 | 0.8189 | 0.7895 | 0.8505 | 0.8447 | 94 |

> [!IMPORTANT]
> **Phát hiện quan trọng**: Gemma-4B đạt điểm **cao nhất** ở tất cả các chỉ số tổng thể, với Macro-F1 vượt trội (+7.85pp so với DeepSeek, +7.90pp so với GPT-OSS). Điều này **bác bỏ** giả thuyết rằng Gemma bị suy giảm hiệu năng do khâu trích xuất ngôn ngữ kém.

### 2.2 Tổng Số Lỗi Trích Xuất

| Mô hình | Tổng lỗi | Lỗi/mẫu |
|--|:--:|:--:|
| **Gemma-4B** | **42** | **0.45** |
| GPT-OSS-20B | 63 | 0.73 |
| DeepSeek-V3.2 | 73 | 0.78 |

Gemma-4B có ít lỗi trích xuất nhất (42 lỗi), ít hơn 42.5% so với DeepSeek (73 lỗi) và 33.3% so với GPT-OSS (63 lỗi).

--

## 3. Phân Tích Chi Tiết Theo Từng Vị Tự

### 3.1 Bảng F1 Theo Vị Tự

| Vị tự | DeepSeek-V3.2 | GPT-OSS-20B | Gemma-4B | Best |
|--|:--:|:--:|:--:|:--:|
| Oral Assertion (P₁) | 0.8972 | **0.9608** | 0.9565 | GPT-OSS |
| Written Assertion (P₂) | 0.2857 | 0.1667 | **0.3750** | Gemma |
| Non-verbal Assertive Conduct (P₃) | 0.5926 | 0.6429 | **0.6667** | Gemma |
| Out of Court (P₄) | 0.9104 | 0.9167 | **0.9760** | Gemma |
| Introduced to Prove Truth (P₅) | 0.7603 | 0.7563 | **0.8644** | Gemma |

> [!TIP]
> Gemma-4B dẫn đầu ở **4/5 vị tự**, chỉ thua GPT-OSS rất nhẹ ở Oral Assertion (0.9565 vs 0.9608, chênh 0.43pp).

### 3.2 Phân Tích Precision / Recall Theo Vị Tự

#### P₁: Oral Assertion

| Mô hình | Precision | Recall | F1 | TP | FP | FN | TN |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V3.2 | **1.0000** | 0.8136 | 0.8972 | 48 | 0 | 11 | 35 |
| GPT-OSS-20B | **1.0000** | 0.9245 | **0.9608** | 49 | 0 | 4 | 33 |
| Gemma-4B | 0.9821 | 0.9322 | 0.9565 | 55 | 1 | 4 | 34 |

- DeepSeek và GPT-OSS có Precision hoàn hảo (1.0) nhưng DeepSeek bỏ sót nhiều (11 FN).
- Gemma đạt cân bằng P/R tốt nhất với chỉ 1 FP.

#### P₂: Written Assertion (Vị tự khó nhất)

| Mô hình | Precision | Recall | F1 | TP | FP | FN | TN |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V3.2 | 0.1818 | 0.6667 | 0.2857 | 2 | 9 | 1 | 82 |
| GPT-OSS-20B | 0.1000 | 0.5000 | 0.1667 | 1 | 9 | 1 | 75 |
| Gemma-4B | 0.2308 | **1.0000** | **0.3750** | 3 | 10 | 0 | 81 |

> [!WARNING]
> Written Assertion là vị tự khó nhất cho tất cả các mô hình (F1 < 0.38). Tất cả đều có tỷ lệ False Positive rất cao (9-10 FP). Gemma là mô hình duy nhất đạt Recall 1.0 (không bỏ sót trường hợp nào) nhưng Precision vẫn rất thấp.

#### P₃: Non-verbal Assertive Conduct

| Mô hình | Precision | Recall | F1 | TP | FP | FN | TN |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V3.2 | 0.7273 | 0.5000 | 0.5926 | 8 | 3 | 8 | 75 |
| GPT-OSS-20B | 0.7500 | 0.5625 | 0.6429 | 9 | 3 | 7 | 67 |
| Gemma-4B | **1.0000** | 0.5000 | 0.6667 | 8 | 0 | 8 | 78 |

- Gemma đạt Precision hoàn hảo (1.0, 0 FP) nhưng Recall thấp giống DeepSeek.
- Tất cả bỏ sót ~50% trường hợp non-verbal assertive conduct.

#### P₄: Out of Court

| Mô hình | Precision | Recall | F1 | TP | FP | FN | TN |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V3.2 | 0.8356 | **1.0000** | 0.9104 | 61 | 12 | 0 | 21 |
| GPT-OSS-20B | 0.8594 | 0.9821 | 0.9167 | 55 | 9 | 1 | 21 |
| Gemma-4B | **0.9531** | **1.0000** | **0.9760** | 61 | 3 | 0 | 30 |

- Gemma vượt trội với chỉ 3 FP so với 12 (DeepSeek) và 9 (GPT-OSS).
- DeepSeek và Gemma đều đạt Recall hoàn hảo (1.0).

#### P₅: Introduced to Prove Truth

| Mô hình | Precision | Recall | F1 | TP | FP | FN | TN |
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-V3.2 | 0.6970 | 0.8364 | 0.7603 | 46 | 20 | 9 | 19 |
| GPT-OSS-20B | 0.6618 | 0.8824 | 0.7563 | 45 | 23 | 6 | 12 |
| Gemma-4B | **0.8095** | **0.9273** | **0.8644** | 51 | 12 | 4 | 27 |

- Gemma dẫn đầu cả Precision lẫn Recall.
- GPT-OSS có nhiều FP nhất (23), cho thấy xu hướng dự đoán dương tính quá mức.

--

## 4. Phân Tích Theo Nhóm Tình Huống (Slice Analysis)

### 4.1 Macro-F1 Theo Slice

| Slice | N | DeepSeek-V3.2 | GPT-OSS-20B | Gemma-4B | Best |
|--|:--:|:--:|:--:|:--:|:--:|
| Non-assertive conduct | 19/18 | 0.0000 | 0.0000 | 0.0000 | Tie |
| Non-verbal hearsay | 12 | 0.4818 | 0.5176 | 0.5000 | GPT-OSS |
| Not introduced to prove truth | 20/17 | 0.5216 | 0.5273 | **0.8000** | **Gemma** |
| Standard hearsay | 29/27 | **0.7962** | 0.7882 | 0.7929 | DeepSeek |
| Statement made in-court | 14/12 | 0.4248 | 0.5028 | **0.5513** | **Gemma** |

### 4.2 Nhận Xét Theo Slice

**Non-assertive conduct (Macro-F1 = 0.0 cho tất cả):**
- Đây là nhóm mà tất cả vị tự đều FALSE trong ground truth → không có True Positive nào, F1 = 0 là đúng toán học. Các mô hình đều nhận diện chính xác (ít FP) nhưng F1 undefined/0 vì Support = 0.

**Not introduced to prove truth:**
- Gemma vượt trội rõ rệt (0.8000 vs ~0.52 cho 2 mô hình kia), cho thấy khả năng phân biệt tốt hơn khi vị tự P₅ = FALSE.

**Statement made in-court:**
- Đây là slice khó thứ hai. Gemma dẫn đầu nhưng vẫn ở mức F1 trung bình (0.5513).

--

## 5. Phân Tích Lỗi Chi Tiết

### 5.1 Phân Bố Lỗi Theo Loại

| Mô hình | Vị tự | False Positive | False Negative | Tổng |
|--|--|:--:|:--:|:--:|
| DeepSeek-V3.2 | Oral Assertion | 0 | 11 | 11 |
| DeepSeek-V3.2 | Written Assertion | 9 | 1 | 10 |
| DeepSeek-V3.2 | Non-verbal Assertive Conduct | 3 | 8 | 11 |
| DeepSeek-V3.2 | Out of Court | 12 | 0 | 12 |
| DeepSeek-V3.2 | Introduced to Prove Truth | 20 | 9 | 29 |
| | | | **Tổng DeepSeek** | **73** |
| GPT-OSS-20B | Oral Assertion | 0 | 4 | 4 |
| GPT-OSS-20B | Written Assertion | 9 | 1 | 10 |
| GPT-OSS-20B | Non-verbal Assertive Conduct | 3 | 7 | 10 |
| GPT-OSS-20B | Out of Court | 9 | 1 | 10 |
| GPT-OSS-20B | Introduced to Prove Truth | 23 | 6 | 29 |
| | | | **Tổng GPT-OSS** | **63** |
| Gemma-4B | Oral Assertion | 1 | 4 | 5 |
| Gemma-4B | Written Assertion | 10 | 0 | 10 |
| Gemma-4B | Non-verbal Assertive Conduct | 0 | 8 | 8 |
| Gemma-4B | Out of Court | 3 | 0 | 3 |
| Gemma-4B | Introduced to Prove Truth | 12 | 4 | 16 |
| | | | **Tổng Gemma** | **42** |

### 5.2 Đặc Điểm Lỗi Của Từng Mô Hình

```
                    Lỗi FP        Lỗi FN        Tổng
DeepSeek-V3.2:        44            29            73
GPT-OSS-20B:          44            19            63
Gemma-4B:             26            16            42
```

| Đặc điểm | DeepSeek-V3.2 | GPT-OSS-20B | Gemma-4B |
|--|--|--|--|
| Xu hướng chính | FP cao ở P₄, P₅ | FP rất cao ở P₅ | FP thấp nhất, FN thấp nhất |
| Điểm yếu lớn nhất | P₅ (29 lỗi) | P₅ (29 lỗi) | P₅ (16 lỗi) |
| Điểm mạnh | P₁ Precision = 1.0 | P₁ Precision = 1.0 | P₃ Precision = 1.0, P₄ chỉ 3 FP |
| Bias | Over-predict P₄, P₅ | Over-predict P₅ mạnh | Cân bằng nhất |

--

## 6. Kết Luận & Hàm Ý

### 6.1 Xếp Hạng Khả Năng Trích Xuất Vị Tự

```
🥇 Gemma-4B       : Macro-F1 = 0.7677 | Micro-F1 = 0.8945 | 42 lỗi
🥈 GPT-OSS-20B    : Macro-F1 = 0.6887 | Micro-F1 = 0.8346 | 63 lỗi  
🥉 DeepSeek-V3.2  : Macro-F1 = 0.6892 | Micro-F1 = 0.8189 | 73 lỗi
```

### 6.2 Trả Lời Câu Hỏi Nghiên Cứu

> **Câu hỏi**: Nếu F1 trích xuất vị tự của Gemma-4B thấp hơn DeepSeek, đó là bằng chứng thực nghiệm khẳng định Gemma bị suy giảm hiệu năng do khâu trích xuất ngôn ngữ kém.

> [!CAUTION]
> **Kết quả thực nghiệm BÁC BỎ giả thuyết trên.** Gemma-4B đạt F1 trích xuất vị tự **CAO NHẤT** trong cả 3 mô hình:
> - Macro-F1: 0.7677 (vs. DeepSeek 0.6892, GPT-OSS 0.6887)
> - Micro-F1: 0.8945 (vs. DeepSeek 0.8189, GPT-OSS 0.8346)
> - Ít lỗi nhất: 42 (vs. DeepSeek 73, GPT-OSS 63)
>
> **Kết luận**: Nếu Gemma-4B có hiệu năng PYTHEN cuối cùng kém hơn DeepSeek/GPT-OSS, nguyên nhân **KHÔNG** nằm ở khâu trích xuất ngôn ngữ. Nguyên nhân suy giảm phải được tìm kiếm ở các khâu khác trong pipeline PYTHEN (ví dụ: suy luận logic, kết hợp vị tự, hoặc quyết định cuối cùng).

### 6.3 Nhận Định Bổ Sung

1. **Written Assertion là điểm mù chung**: Tất cả 3 mô hình đều gặp khó khăn lớn ở vị tự này (F1 max = 0.375). Cần xem xét cải thiện prompt hoặc bổ sung ví dụ few-shot.

2. **Non-verbal Assertive Conduct**: Recall chỉ đạt ~50% cho tất cả mô hình, cho thấy đây là khái niệm pháp lý khó nhận diện.

3. **Introduced to Prove Truth**: Vị tự gây nhiều lỗi nhất cho tất cả mô hình, đặc biệt là FP. Mô hình có xu hướng dự đoán dương tính quá mức cho vị tự này.

--

## 7. Tệp Đầu Ra

| Tệp | Mô tả |
|--|--|
| [`evaluate_extraction.py`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/evaluate_extraction.py) | Script đánh giá chính |
| [`overall_metrics.csv`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/overall_metrics.csv) | Chỉ số tổng hợp (Macro/Micro F1, Accuracy) |
| [`per_predicate_metrics.csv`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/per_predicate_metrics.csv) | Chi tiết P/R/F1 cho từng vị tự × mô hình |
| [`per_slice_metrics.csv`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/per_slice_metrics.csv) | Phân tích theo nhóm tình huống (slice) |
| [`error_summary.csv`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/error_summary.csv) | Tổng hợp FP/FN theo vị tự × mô hình |
| [`extraction_errors_detail.csv`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/extraction_errors_detail.csv) | Chi tiết từng lỗi trích xuất (row-level) |
| [`evaluation_results.json`](file:///Users/admin/Desktop/PYTHEN_Paper_Experiments/ablation_study/predicate_extraction/evaluation_results.json) | Toàn bộ kết quả ở dạng JSON |

--

*Báo cáo được tạo tự động bởi `evaluate_extraction.py` — Ablation Study Step 3*
*Ngày: 2026-08-21*
