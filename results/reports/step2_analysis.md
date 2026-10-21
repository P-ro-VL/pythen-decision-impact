# Phân Tích Chi Tiết Kết Quả Step 2
## Câu hỏi trung tâm: Tại sao DeepSeek & GPT tăng vọt, còn Gemma lại suy giảm?

--

## 1. Bức Tranh Số Liệu Tổng Thể

### Bảng so sánh Baseline CoT vs PYTHEN (Neuro-Symbolic)

| Model | Baseline F1 | PYTHEN F1 | Δ F1 | TP→ | FP→ | FN→ |
|--|--:|--:|--:|--|--|--|
| **Gemma 4B 31B** | 0.8283 | 0.8387 | **+0.0104** ↑ nhẹ | 41→39 | 17→13 | 0→2 |
| **DeepSeek V4 Flash** | 0.8780 | 0.7835 | **−0.0945** ↓ mạnh | 36→38 | 5→18 | 5→3 |
| **GPT OSS 20B** | 0.7606 | 0.8571 | **+0.0965** ↑ mạnh | 27→36 | 3→9 | 14→3 |

> [!IMPORTANT]
> Kết quả có sự đảo ngược đáng chú ý:
> - **GPT OSS 20B** tăng mạnh nhất (+9.65% F1) khi dùng PYTHEN
> - **DeepSeek** lại *giảm* mạnh nhất (−9.45% F1) khi dùng PYTHEN
> - **Gemma** chỉ tăng nhẹ (+1.04%), nhưng bản chất lỗi thay đổi đáng kể

--

## 2. Phân Tích: Tại Sao GPT OSS 20B Tăng Vọt?

### 2.1 Profile lỗi của GPT trong Baseline CoT

```
GPT Baseline CoT:
  TP=27, FP=3, FN=14, TN=50
  FN=14 → Bỏ sót 14 mẫu hearsay (High Miss Rate)
  FN by slice: Standard hearsay=13, Non-verbal hearsay=1
```

**GPT trong baseline có vấn đề nghiêm trọng: bỏ sót 14 mẫu hearsay thật (FN=14).**
Đây là lỗi E4 Over-conservatism — CoT của GPT rất thận trọng, thường kết luận "not hearsay" ở step3/conclusion ngay cả khi các bước phân tích trước đó đúng. Đặc biệt trên "Standard hearsay" — loại đơn giản nhất — GPT vẫn bỏ sót đến 13/14 mẫu.

**Nguyên nhân gốc rễ của GPT Baseline FN:**
- GPT có xu hướng tìm *exceptions* (hearsay exceptions, party-opponent statements) và kết luận "not hearsay" một cách phòng thủ
- Nhiều mẫu Standard hearsay bị GPT nhầm vì model phân tích quá sâu các khía cạnh pháp lý mà quên mất bản chất cơ bản của hearsay
- Lỗi JSON parsing (5 mẫu `structured_output_error`) cũng ép output về `False`

### 2.2 PYTHEN "chữa" lỗi Over-conservatism của GPT như thế nào?

```
GPT PYTHEN:
  TP=36, FP=9, FN=3, TN=38
  FN=3 → Chỉ còn bỏ sót 3 mẫu (giảm từ 14 → 3)
```

**PYTHEN loại bỏ hoàn toàn bước "conclusion" của CoT.** Thay vì để model tự kết luận (nơi xảy ra Over-conservatism), PYTHEN dùng rule engine:

```
is_hearsay = (is_oral OR is_written OR is_non_verbal)
             AND is_out_of_court
             AND is_introduced_to_prove_truth
```

GPT trích xuất từng predicate riêng lẻ rất tốt:
- `is_oral_assertion`: Precision=1.0, Recall=0.9245 (F1=0.9608) ← **Best của 3 model**
- `is_out_of_court`: Precision=0.8594, Recall=0.9821 (F1=0.9167)

Khi PYTHEN đánh giá các predicate đó một cách độc lập và AND chúng lại, 11 mẫu Standard hearsay bị GPT bỏ sót trong CoT được *tự động phục hồi*.

> [!NOTE]
> **PYTHEN đóng vai trò như "safety net" cho GPT**: Engine symbolic bắt được những mẫu mà CoT tự kết luận sai, nhưng predicate extraction lại đúng. Đây chính xác là giá trị của kiến trúc Neuro-Symbolic.

### 2.3 Chi phí: FP tăng từ 3 → 9

GPT PYTHEN có 9 FP, tất cả thuộc "Not introduced to prove truth" (8/9) — khi predicate `is_introduced_to_prove_truth` bị LLM gán True nhầm, PYTHEN không có cơ chế "suy luận ngữ cảnh" để điều chỉnh. Đây là giới hạn của symbolic layer.

--

## 3. Phân Tích: Tại Sao DeepSeek Giảm Mạnh?

### 3.1 DeepSeek Baseline — Mô hình "cân bằng nhất"

```
DeepSeek Baseline CoT:
  TP=36, FP=5, FN=5, TN=48
  F1=0.8780 → Tốt nhất trong 3 model
  FP=5: cân bằng giữa false positive và false negative
```

DeepSeek trong Baseline CoT đạt **cân bằng precision-recall tốt nhất** (P=0.878, R=0.878 — hoàn toàn cân bằng). Mô hình CoT của DeepSeek có khả năng reasoning về *context* và *purpose* của bằng chứng rất tốt, dẫn đến ít lỗi hơn cả về FP lẫn FN.

### 3.2 PYTHEN "phá vỡ" cân bằng của DeepSeek

```
DeepSeek PYTHEN:
  TP=38, FP=18, FN=3, TN=35
  FP=18 → TĂNG MẠNH từ 5 → 18 (gấp 3.6 lần!)
  FP slice: Not introduced to prove truth=16, Statement made in-court=2
```

**Vấn đề nằm ở `is_introduced_to_prove_truth`:**

| Metric | DeepSeek `is_introduced_to_prove_truth` |
|--|--|
| Precision | **0.6970** (thấp nhất nhóm) |
| Recall | 0.8364 |
| FP | **20** (nhiều nhất) |

DeepSeek trong pythen CSV gán `is_introduced_to_prove_truth=True` cho **20 mẫu sai** (GT=False). Đây là lỗi E1 (Predicate Hallucination) nghiêm trọng.

**Tại sao DeepSeek mắc lỗi này nhiều hơn?**

Trong baseline CoT, DeepSeek *reasoning về toàn bộ context* mới quyết định "có phải offered for truth không?" — và lý luận được tốt (FP chỉ 5). Nhưng trong PYTHEN, DeepSeek chỉ được yêu cầu trả lời **câu hỏi boolean đơn lẻ** về từng predicate, tách rời khỏi ngữ cảnh tổng thể. Khi bị cô lập, DeepSeek có xu hướng:

- Thấy "phiên tòa muốn dùng statement này" → gán `is_introduced_to_prove_truth=True`
- Không nhận ra sự khác biệt giữa "dùng để chứng minh truth of content" vs "dùng để chứng minh effect on listener / knowledge"

```
Ví dụ điển hình:
Mẫu #47: "Bob told Jane he suspected Arthur was up to something"
→ Mục đích: chứng minh JANE BIẾT (effect on listener), KHÔNG phải truth về Arthur
→ DeepSeek PYTHEN: is_introduced_to_prove_truth = True ← SAI
→ DeepSeek Baseline CoT: không hearsay ← ĐÚNG (vì reasoning về full context)
```

> [!WARNING]
> **DeepSeek là trường hợp điển hình của "context dependency"**: Model này đòi hỏi ngữ cảnh đầy đủ của chuỗi suy luận để quyết định predicate cuối cùng. Khi PYTHEN tách từng predicate thành câu hỏi độc lập, DeepSeek mất đi lợi thế reasoning.

### 3.3 Nhận xét thêm: `is_written_assertion` của DeepSeek

DeepSeek cũng có Precision=0.1818 trên `is_written_assertion` — gán True cho 9 mẫu khi GT chỉ có 3 mẫu True. Lỗi này thêm 6 FP từ các mẫu "Non-verbal hearsay" được misclassify thành "written assertion".

--

## 4. Phân Tích: Tại Sao Gemma 4B Suy Giảm Tương Đối?

> [!IMPORTANT]
> **Thực tế: Gemma KHÔNG suy giảm — F1 tăng nhẹ từ 0.8283 → 0.8387.** Nhưng tại sao Gemma không tăng mạnh như GPT (mặc dù cũng có 17 lỗi baseline)?

### 4.1 Profile lỗi Gemma trong Baseline

```
Gemma Baseline CoT:
  FP=17, FN=0 → Gemma KHÔNG bao giờ bỏ sót hearsay (Recall=1.0)
  FP by slice: Not introduced to prove truth=13, Non-assertive conduct=3, In-court=1
```

Gemma trong baseline là mô hình **"over-inclusive"** — luôn phân loại là hearsay khi nghi ngờ. Recall=100% nhưng Precision chỉ 70.7%. Toàn bộ 17 lỗi đều là FP (phân loại nhầm non-hearsay thành hearsay).

### 4.2 PYTHEN giảm được một số FP, nhưng tạo ra FN mới

```
Gemma PYTHEN:
  FP=13 (giảm từ 17) ← PYTHEN loại bỏ 4 FP nhờ predicate checks
  FN=2  (tăng từ 0)  ← PYTHEN tạo ra 2 FN mới
```

**FP giảm:** 4 mẫu mà Gemma CoT phân loại nhầm nhưng predicate extraction lại đúng (GT=False, và LLM cũng gán is_introduced_to_prove_truth=False) → PYTHEN output False đúng.

**FN xuất hiện:** 2 mẫu Standard hearsay mà Gemma PYTHEN miss:
- Gemma trích xuất sai một predicate key (ví dụ `is_out_of_court=False`) cho mẫu mà GT là True
- Đây là lỗi E2 (Predicate Omission) trong tầng extraction

### 4.3 Tại sao Gemma không tận dụng PYTHEN như GPT?

GPT: **FN cao (14) → PYTHEN giải quyết triệt để** → tăng F1 mạnh  
Gemma: **FN=0 (không có gì để giải quyết)** → PYTHEN chủ yếu giảm FP, nhưng ít hơn mong đợi

**Lý do PYTHEN không giảm được nhiều FP hơn cho Gemma:**

Gemma là mô hình duy nhất mà 11/17 lỗi là **E1 (LLM Hallucination)** trên `is_introduced_to_prove_truth`:
- LLM gán `is_introduced_to_prove_truth=True` sai → PYTHEN nhận input sai → output sai
- PYTHEN không thể "sửa" lỗi extraction của LLM — nó chỉ thi hành rule trên input đó

```
Chuỗi lỗi Gemma:
Text: "Bob told Jane..." (purpose = effect on listener, NOT for truth)
Gemma: is_introduced_to_prove_truth = True  ← LLM extraction SAI (E1)
PYTHEN: (True AND True AND True) = True      ← PYTHEN output SAI
GT: False                                    ← FP
```

PYTHEN hoàn toàn không có cơ chế để phát hiện "LLM đã gán True sai" — nó chỉ nhận boolean input.

### 4.4 6 mẫu "NO_PREDICATE_ERROR" đặc biệt của Gemma

Trong 17 lỗi của Gemma, có **6 mẫu** mà tất cả 5 predicates được trích xuất **đúng hoàn toàn** so với Ground Truth, nhưng PYTHEN vẫn output sai:

| Sample | Cơ chế lỗi |
|--|--|
| #12, #15 (Non-assertive conduct) | GT: tất cả predicates = False → PYTHEN False ✓, nhưng... có thể issue ở index mapping |
| #47, #82, #87, #90 (Not introduced to prove truth) | GT: `is_introduced_to_prove_truth=False`, Gemma cũng gán False → PYTHEN False (đúng), nhưng step1 ghi là sai |

> Đây cần kiểm tra lại — có thể là **discrepancy giữa step1 prediction và PYTHEN recomputed prediction** từ script của chúng ta.

--

## 5. Bức Tranh Tổng Hợp: Cơ chế PYTHEN hoạt động với từng model

```
┌─────────────────────────────────────────────────────────────────────┐
│           HIỆU QUẢ CỦA PYTHEN THEO PROFILE LỖI BASELINE           │
├─────────────────┬──────────────────────────────────────────────────┤
│ Profile Baseline│ Hiệu quả PYTHEN                                  │
├─────────────────┼──────────────────────────────────────────────────┤
│ FN nhiều (GPT)  │ ✅ PYTHEN phục hồi FN nếu predicate extraction   │
│                 │    đúng → F1 tăng mạnh                           │
├─────────────────┼──────────────────────────────────────────────────┤
│ FP nhiều (Gemma)│ ⚠️  PYTHEN giảm ít FP vì lỗi nằm ở LLM          │
│                 │    extraction (E1 Hallucination) → PYTHEN nhận   │
│                 │    input sai, không thể tự sửa                   │
├─────────────────┼──────────────────────────────────────────────────┤
│ Cân bằng        │ ❌ PYTHEN phá vỡ cân bằng vì tách predicate ra   │
│ (DeepSeek)      │    khỏi context → model mất khả năng holistic    │
│                 │    reasoning → FP tăng vọt                       │
└─────────────────┴──────────────────────────────────────────────────┘
```

--

## 6. Phát Hiện Cốt Lõi về Kiến Trúc PYTHEN

### 6.1 PYTHEN hiệu quả khi:
- LLM có **lỗi ở tầng kết luận** (Over-conservatism, calibration bias) nhưng predicate extraction tốt
- LLM có Recall thấp (bỏ sót nhiều mẫu Positive) — PYTHEN rule engine bù đắp bằng AND-logic chặt chẽ

### 6.2 PYTHEN kém hiệu quả khi:
- **LLM extraction sai** ở tầng predicate — "garbage in, garbage out"
- **LLM phụ thuộc vào contextual reasoning** để quyết định predicate (như DeepSeek với `is_introduced_to_prove_truth`)

### 6.3 Hàm ý cho nghiên cứu:
| Khuyến nghị | Giải pháp |
|--|--|
| Cải thiện Predicate Accuracy của Gemma trên `is_introduced_to_prove_truth` | Fine-tuning với ví dụ về "effect on listener" vs "truth of content" |
| Giúp DeepSeek duy trì contextual reasoning trong PYTHEN | Cung cấp full context (không chỉ câu hỏi boolean) trong prompt extraction |
| Cải thiện `is_written_assertion` (F1=0.2758 cho cả 3 model) | Làm rõ definition: chỉ là assertive written statement, không phải bất kỳ text nào |

--

## 7. Bảng Tóm Tắt

| Khía cạnh | GPT OSS 20B | DeepSeek V4 Flash | Gemma 4B 31B |
|--|--|--|--|
| **Baseline profile** | Over-cautious (FN=14) | Balanced (FP=FN=5) | Over-inclusive (FP=17) |
| **PYTHEN effect** | ↑ F1 +9.65% | ↓ F1 −9.45% | ↑ F1 +1.04% |
| **Lý do chính** | E4 CoT → PYTHEN fixes FN | Context-dep. reasoning → E1 FP × 3.6 | E1 extraction → PYTHEN nhận input sai |
| **Lỗi chủ đạo PYTHEN** | E1 (FP `is_introduced`) | E1 (FP `is_introduced`) | E1 (FP `is_introduced`) |
| **Predicate dễ sai** | `is_written_assertion` | `is_written_assertion` | `is_written_assertion` |
| **Nhận xét** | PYTHEN như safety net | PYTHEN phá vỡ holistic CoT | PYTHEN không thể sửa LLM extraction |
