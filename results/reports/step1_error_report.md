# Step 1 — Báo Cáo Thu Thập & Trích Xuất Dữ Liệu Lỗi
**Bộ dữ liệu:** Hearsay (94 mẫu) | **Ngày thực hiện:** 20/08/2026

--

## 1. Tổng Quan Nhiệm Vụ

Bước 1 thu thập và trích xuất toàn bộ dữ liệu lỗi trên bộ dữ liệu **Hearsay (94 mẫu)** cho **3 mô hình** với **chế độ Neuro-Symbolic (PYTHEN / Ours)**:

| Mô hình | File CSV | Ghi chú |
|--|--|--|
| DeepSeek V4 Flash | `deepseek.deepseek-v4-flash.csv` | |
| GPT OSS 20B | `openai.gpt-5.6-luna.csv` | |
| Gemma 4 31B | `google.gemma-4-31b.csv` | Mô hình trọng tâm |

> **Lưu ý về chế độ:** Mỗi file CSV chứa kết quả PYTHEN (Neuro-Symbolic). Trường `thinking_process` lưu CoT reasoning (suy luận từng bước), trường `pythen_prediction` lưu nhãn dự đoán cuối cùng của hệ thống NS.

--

## 2. Kết Quả Tổng Hợp Theo Mô Hình

### 2.1 Bảng Thống Kê Chính

| Mô hình | Tổng mẫu | NS Đúng | NS Sai | Có CoT Reasoning | NS Sai & Có CoT |
|--|:--:|:--:|:--:|:--:|:--:|
| **DeepSeek V4 Flash** | 94 | 84 | **10** | 94 | 0 |
| **GPT OSS 20B** | 94 | 77 | **17** | 94 | 17 |
| **Gemma 4 31B** | 94 | 77 | **17** | 94 | **17** ⚠️ |

> **Tổng lỗi NS:** 44 lỗi trên 3 mô hình (10 + 17 + 17)  
> **Lỗi trong danh sách output:** 34 lỗi (loại bỏ các trường hợp `pred_ns = None`)

### 2.2 Tỉ Lệ Lỗi NS (Error Rate)

```
DeepSeek V4 Flash :  10/94 = 10.6%
GPT OSS 20B       :  17/94 = 18.1%
Gemma 4 31B       :  17/94 = 18.1%
```

--

## 3. Phân Loại Lỗi

### 3.1 Phân Loại Theo Loại Lỗi

Toàn bộ 34 lỗi được phân loại theo cấu trúc:

| Loại lỗi | Mô hình | Số lượng |
|--|--|:--:|
| `NS_WRONG_WITH_COT` | GPT OSS 20B | 17 |
| `NS_WRONG_WITH_COT` | Gemma 4 31B | 17 |
| *(DeepSeek: không có mẫu nào thuộc NS_WRONG_WITH_COT)* | DeepSeek V4 Flash | 0 |

> **Định nghĩa `NS_WRONG_WITH_COT`:** Mẫu mà hệ thống Neuro-Symbolic dự đoán **sai** trong khi vẫn tồn tại CoT reasoning (thinking_process). Đây là nhóm quan trọng nhất để phân tích vì cho thấy mô hình *có thực hiện suy luận* nhưng vẫn đưa ra kết luận sai.

### 3.2 Phân Loại Lỗi Theo Slice (Danh mục bằng chứng)

Dựa trên **34 lỗi** trong `step1_all_errors.csv`:

| Slice | GPT OSS 20B | Gemma 4 31B | Tổng |
|--|:--:|:--:|:--:|
| `Standard hearsay` | 13 | 0 | 13 |
| `Not introduced to prove truth` | 4 | 12 | 16 |
| `Non-assertive conduct` | 0 | 3 | 3 |
| `Statement made in-court` | 1 | 1 | 2 |
| `Non-verbal hearsay` | 1 | 0 | 1 |

--

## 4. Nhóm Trọng Tâm: Gemma 4 31B — NS Sai Có CoT Reasoning

Đây là nhóm chính gây tụt F1-score của Gemma 4 khi chuyển sang hệ thống Neuro-Symbolic. Cả **17 mẫu** đều có pattern: `GT = False` nhưng `NS Predicted = True` (mô hình phân loại sai là Hearsay trong khi Ground Truth là không phải Hearsay).

### 4.1 Danh Sách 17 Mẫu Trọng Tâm

| idx | Ground Truth | NS Predicted | Slice | Nội dung tóm tắt |
|:--:|:--:|:--:|--|--|
| 6 | False | True | Non-assertive conduct | Patricia có poster lyrics "Viva la Vida" → fan Coldplay? |
| 12 | False | True | Non-assertive conduct | Jessica được chẩn đoán cận thị → biết mình đang xâm phạm? |
| 15 | False | True | Non-assertive conduct | Karen thi bằng lái xe nhiều lần → lái xe cẩu thả? |
| 23 | False | True | Statement made in-court | Daniel làm chứng tại tòa rằng đã nói với vợ anh ta mệt |
| 47 | False | True | Not introduced to prove truth | Bob nói với Jane "Arthur đang làm gì đó khuất tất" |
| 72 | False | True | Not introduced to prove truth | Công nhân cầm biển hiệu đòi lương công bằng |
| 74 | False | True | Not introduced to prove truth | Tom chỉ tay vào bị cáo tại hiện trường |
| 76 | False | True | Not introduced to prove truth | Vincent được bạn nói Roy vừa tấn công anh ta |
| 78 | False | True | Not introduced to prove truth | Lori tuyên bố công khai "Tôi chưa bao giờ thấy Mason uống rượu" |
| 79 | False | True | Not introduced to prove truth | Gerald nói "Tôi bị Kathryn tấn công" (ngay sau sự việc) |
| 80 | False | True | Not introduced to prove truth | Dylan nói "Andrew là kẻ nói dối và đạo đức giả" |
| 82 | False | True | Not introduced to prove truth | Patty nói với Arthur "Dave thực sự không trung thực" |
| 85 | False | True | Not introduced to prove truth | Mary nói với nhân viên mall cô dự định mua kim cương |
| 86 | False | True | Not introduced to prove truth | Anne nói David là sếp tốt nhất thế giới |
| 87 | False | True | Not introduced to prove truth | Daniel thốt lên "thế giới đang sụp đổ" khi nghe tin |
| 90 | False | True | Not introduced to prove truth | Khách hàng gọi nhầm nhà hàng do trademark |
| 91 | False | True | Not introduced to prove truth | Michael nói patent "được viết kém" |

### 4.2 Phân Tích Pattern Lỗi Của Gemma 4

#### Pattern 1: Lỗi trên "Not introduced to prove truth" (12/17 mẫu — 70.6%)

Đây là nhóm lỗi chủ đạo. Gemma 4 nhận ra đúng rằng:
- ✅ Có một **statement** (lời nói/hành vi assertive)
- ✅ Statement được thực hiện **ngoài phiên tòa** (out-of-court)
- ❌ **Sai ở Step 3**: Kết luận statement được dùng để chứng minh **truth of the matter asserted** — nhưng Ground Truth là **không** (chúng được dùng để chứng minh *kiến thức*, *trạng thái tinh thần*, *sự nhầm lẫn*, v.v.)

**Ví dụ điển hình (idx=91):**
```
Mẫu: To prove that Michael knew of the existing patent, 
      the fact that he told his friend that the patent was poorly written.

Gemma reasoning step3: "Yes, the statement is being introduced to prove 
  that Michael knew of the existing patent. By asserting that the patent 
  was 'poorly written,' the statement is used to prove the truth of the 
  underlying fact that Michael had seen and was aware of the patent."

→ Gemma kết luận: IS Hearsay (True)  ← SAI
→ Ground Truth:   NOT Hearsay (False) — vì dùng để chứng minh KNOWLEDGE, 
                                         không phải truth of content của statement
```

**Ví dụ điển hình (idx=47):**
```
Mẫu: On the issue of whether Jane knew that Arthur was having an affair, 
      the fact that Bob told Jane he suspected Arthur was up to something.

Gemma reasoning step3: "Yes, the evidence is being introduced to establish 
  that Jane had knowledge of Arthur's possible infidelity. To prove Jane 
  'knew' or was alerted to the affair, the court must rely on the truth 
  of Bob's assertion that Arthur was 'up to something'."

→ Gemma kết luận: IS Hearsay (True)  ← SAI
→ Ground Truth:   NOT Hearsay (False) — statement dùng để chứng minh 
                                         EFFECT ON LISTENER (Jane biết),
                                         không phải truth của nội dung statement
```

#### Pattern 2: Lỗi trên "Non-assertive conduct" (3/17 mẫu — 17.6%)

Gemma nhầm lẫn giữa **conduct** và **statement**:

- **idx=6**: Poster lyrics bài hát → Gemma coi là "written assertion" → phân loại là Hearsay. Thực ra, hành vi treo poster là **non-assertive conduct**, không phải statement có chủ đích communicate thông tin.
- **idx=12**: Chẩn đoán y khoa → Gemma coi là "oral/written assertion" → phân loại sai. Thực ra là **hành động y tế**, không phải assertive statement.
- **idx=15**: Hồ sơ thi bằng lái → phân loại sai tương tự (nhầm record hành chính thành assertive statement).

#### Pattern 3: Lỗi trên "Statement made in-court" (1/17 mẫu — 5.9%)

- **idx=23**: Daniel **làm chứng tại tòa** rằng trước đó đã nói với vợ anh ta mệt. Gemma coi statement gốc (nói với vợ) là out-of-court và offered for truth. Nhưng vì Daniel đang **testify in-court** về statement của chính mình → không phải hearsay theo FRE.

--

## 5. Phân Tích So Sánh: GPT OSS 20B vs Gemma 4 31B

| Khía cạnh | GPT OSS 20B | Gemma 4 31B |
|--|--|--|
| Tổng lỗi NS | 17 | 17 |
| Slice lỗi chính | Standard hearsay (13/17) | Not introduced to prove truth (12/17) |
| Lỗi parsing/JSON | 5/17 (structured_output_error) | 0/17 |
| Nhầm lẫn về purpose | Phân loại sai standard cases | Nhầm lẫn nghiêm trọng về mục đích đưa bằng chứng |
| Pattern lỗi CoT | Suy luận sai ở Step 3 | Suy luận sai ở Step 3 (cùng pattern) |

> **Điểm khác biệt:** GPT OSS 20B có **5 lỗi parsing JSON** (structured_output_error) khiến NS output bị `False` mặc dù reasoning có thể đúng. Gemma 4 không có lỗi parsing nhưng reasoning sai nghiêm trọng hơn về mặt pháp lý.

--

## 6. Điểm Đặc Biệt: DeepSeek V4 Flash

DeepSeek có kết quả tốt nhất với chỉ **10 lỗi NS (10.6%)** và **0 mẫu NS_WRONG_WITH_COT**. Điều này cho thấy:
- DeepSeek xử lý tốt hơn việc chuyển đổi CoT reasoning → NS output
- Không có sự không nhất quán giữa reasoning và kết luận
- 10 lỗi DeepSeek là loại khác (có thể do thiếu CoT hoặc cấu trúc khác — cần kiểm tra Step 2)

--

## 7. Tóm Tắt Phát Hiện Chính

> [!IMPORTANT]
> **Kết luận trọng tâm:** Gemma 4 31B có **17/17 lỗi NS đều kèm CoT reasoning**, cho thấy vấn đề không phải ở việc *thiếu suy luận* mà ở việc **suy luận sai về mục đích của bằng chứng** (đặc biệt là Step 3 — xác định có phải offered for truth hay không).

| # | Phát hiện | Mức độ |
|:--:|--|:--:|
| 1 | Gemma 4 nhầm lẫn giữa "offered to prove truth" và "offered to prove knowledge/effect on listener" | 🔴 Nghiêm trọng |
| 2 | Gemma 4 không phân biệt được non-assertive conduct và assertive statement | 🔴 Nghiêm trọng |
| 3 | GPT OSS 20B có lỗi JSON parsing gây sai lệch output dù reasoning có thể đúng | 🟡 Trung bình |
| 4 | DeepSeek V4 Flash thực hiện tốt nhất với error rate thấp nhất (10.6%) | 🟢 Tích cực |
| 5 | Cả 3 mô hình đều có CoT reasoning (94/94 mẫu), không có mẫu nào thiếu reasoning | 🟢 Tích cực |

--

## 8. Output Files

| File | Mô tả | Số dòng |
|--|--|:--:|
| `step1_all_errors.csv` | Danh sách 34 lỗi NS tổng hợp 3 mô hình | 34 |
| `step1_gemma_focus_samples.csv` | 17 mẫu trọng tâm Gemma 4 (NS sai có CoT) | 17 |
| `step1_summary.csv` | Bảng tóm tắt thống kê theo mô hình | 3 |

### Cấu trúc cột trong `step1_all_errors.csv`

| Cột | Ý nghĩa |
|--|--|
| `model` | Tên mô hình |
| `index` | Index mẫu trong dataset (0–93) |
| `ground_truth` | Nhãn đúng (True/False) |
| `pred_ns` | Nhãn dự đoán của Neuro-Symbolic |
| `error_type` | Loại lỗi (`NS_WRONG_WITH_COT` / `NS_WRONG_NO_COT`) |
| `slice` | Danh mục bằng chứng pháp lý |
| `text` | Nội dung câu hỏi |
| `raw_cot_output` | CoT reasoning nguyên bản (thinking_process) |
| `extracted_json_ns` | JSON output của hệ thống NS (llm_output) |
| `is_focus_sample` | True nếu là mẫu trọng tâm Gemma |

--

## 9. Bước Tiếp Theo (Step 2)

Dựa trên kết quả Step 1, Step 2 sẽ tập trung **phân tích sâu CoT reasoning** của 17 mẫu trọng tâm Gemma 4:

1. **Phân tích Step 3** trong reasoning → xác định cụ thể điểm suy luận sai
2. **So sánh** reasoning của Gemma với GPT OSS 20B trên cùng mẫu (đặc biệt idx=72, 74)
3. **Phân loại** nguyên nhân lỗi theo taxonomy: *Purpose confusion* / *Conduct vs Statement* / *In-court vs Out-of-court*
4. **Tính toán** impact lên F1-score của Gemma 4

--

*Generated by Step 1 error analysis pipeline — PYTHEN Paper Experiments*
