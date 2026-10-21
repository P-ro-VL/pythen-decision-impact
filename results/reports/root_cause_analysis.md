# Phân Tích Nguyên Nhân Lỗi NS — Root Cause Analysis

> **Câu hỏi:** Tại sao DeepSeek chỉ có 10 lỗi trong khi GPT và Gemma đều có 17 lỗi?  
> Và tại sao pattern lỗi của 3 mô hình lại khác nhau hoàn toàn?

--

## TL;DR — Kết Luận Trong 3 Dòng

| Mô hình | Nguyên nhân lỗi chính | Loại sai lệch |
|--|--|--|
| **DeepSeek V4 Flash** | Lỗi suy luận thực sự, **phân bổ đều** giữa FP và FN | Balanced error |
| **GPT OSS 20B** | **JSON parsing failure** (4 lỗi) + suy luận sai trên Standard hearsay (13 lỗi) | Conservative bias (FN chủ đạo) |
| **Gemma 4 31B** | **Systematic overclassification** — coi mọi thứ là Hearsay, **100% FP, 0% FN** | Aggressive bias |

--

## 1. Phân Tích FP vs FN: Bias Của Từng Mô Hình

### Định nghĩa
- **FP (False Positive):** GT = `False` (không phải Hearsay) nhưng mô hình dự đoán `True` (là Hearsay)
- **FN (False Negative):** GT = `True` (là Hearsay) nhưng mô hình dự đoán `False` (không phải Hearsay)

### Bảng FP/FN

| Mô hình | Tổng sai | FP | FN | FP:FN ratio | Bias |
|--|:--:|:--:|:--:|:--:|--|
| **DeepSeek V4 Flash** | 10 | **5** | **5** | 1:1 | Không có bias |
| **GPT OSS 20B** | 17 | **3** | **14** | 1:4.7 | **Conservative** (thiếu nhận Hearsay) |
| **Gemma 4 31B** | 17 | **17** | **0** | ∞ | **Extreme aggressive** (nhận quá mức) |

```
DeepSeek : ████████████████████████████████████████ FP=5  FN=5   (cân bằng)
GPT      : ████████                                 FP=3  ████████████████████████████████████ FN=14
Gemma    : ████████████████████████████████████████████████████████████████████████████████████ FP=17  FN=0
```

> [!IMPORTANT]
> Gemma 4 có **hội chứng overclassification cực đoan**: trong 17 lỗi, **không có một mẫu nào** là FN. Gemma luôn dự đoán **Hearsay = True** cho các mẫu khó, dẫn đến FP = 100%.

--

## 2. Phân Tích Theo Slice: Điểm Yếu Đặc Thù Của Từng Mô Hình

### Accuracy (%) Theo Danh Mục Bằng Chứng

| Slice | n | DeepSeek | GPT OSS 20B | Gemma 4 31B |
|--|:--:|:--:|:--:|:--:|
| **Non-assertive conduct** | 19 | 94.7% | **100%** | 84.2% 🔴 |
| **Non-verbal hearsay** | 12 | 91.7% | 91.7% | **100%** |
| **Not introduced to prove truth** | 20 | 85.0% | 90.0% | **35.0%** 🔴🔴 |
| **Standard hearsay** | 29 | 86.2% | 55.2% 🔴 | **100%** |
| **Statement made in-court** | 14 | 92.9% | 92.9% | 92.9% |

### Đọc Từng Mô Hình

**DeepSeek V4 Flash** — *Generalist, không có điểm yếu cực đoan*
- Hiệu suất đồng đều nhất: 85–95% trên tất cả slice
- Điểm yếu nhỏ nhất: *Not introduced to prove truth* (85%) và *Standard hearsay* (86.2%)
- Không có slice nào bị sập hoàn toàn

**GPT OSS 20B** — *Điểm yếu rõ ràng: Standard hearsay*
- Chỉ đạt **55.2%** trên Standard hearsay (thấp hơn random baseline 50%)
- Các slice khác: 90–100% (rất tốt)
- **13/14 FN** đều từ Standard hearsay → mô hình không nhận ra các trường hợp rõ ràng nhất

**Gemma 4 31B** — *Điểm yếu nghiêm trọng: Not introduced to prove truth*
- Chỉ đạt **35%** trên *Not introduced to prove truth* (tệ hơn random)
- Đạt **100%** trên *Standard hearsay* và *Non-verbal hearsay* — nghịch lý hoàn toàn với GPT
- Bù đắp sai bằng cách overclassify tất cả → đúng trên simple cases, sai trên complex cases

--

## 3. Nguyên Nhân Lỗi GPT OSS 20B: Hai Nguồn Gốc Khác Nhau

GPT có **2 loại nguyên nhân** hoàn toàn khác biệt:

### 3a. Lỗi JSON Parsing (4/17 = 23.5% lỗi)

```
Lỗi: structured_output_error
"Expecting ':' delimiter: line 1 column 639"
"Expecting ',' delimiter: line 1 column 664"
```

GPT sinh ra JSON không hợp lệ → hệ thống NS **fallback về `result: False`** → gây FN giả tạo.

| idx | GT | Pred NS | Lý do thực |
|:--:|:--:|:--:|--|
| 33 | True | False | JSON parsing error |
| 36 | True | False | JSON parsing error |
| 39 | True | False | JSON parsing error |
| 43 | True | False | JSON parsing error |

> [!WARNING]
> **4 trong số 17 lỗi GPT không phải lỗi suy luận** mà là lỗi infrastructure. Nếu fix được JSON parsing, GPT thực chất chỉ có 13 lỗi thực sự (13.8% error rate, tương đương DeepSeek).

### 3b. Lỗi Suy Luận Thực Sự: Standard Hearsay (13/17 = 76.5% lỗi)

GPT nhận ra statement, nhận ra out-of-court, nhưng **sai ở Step 3** theo hướng ngược lại với Gemma:

**Ví dụ idx=37** (GT=True, Pred=False):
```
Stephen nói với vợ qua điện thoại rằng anh không chú ý đến đường.
→ Được đưa ra để chứng minh Stephen lái xe qua đèn đỏ.

GPT Step3: "No. The statement does not assert that Stephen drove 
  through a stoplight. It is offered as circumstantial evidence 
  from which the factfinder may infer..."

→ GPT kết luận: NOT Hearsay ← SAI
→ Ground Truth: IS Hearsay — đây là trường hợp standard hearsay rõ ràng
```

**GPT sai theo hướng:** quá thông minh, áp dụng exception "circumstantial evidence" ngay cả khi không cần thiết. GPT học được pattern "effect on listener" và áp dụng quá rộng sang cả Standard hearsay.

--

## 4. Nguyên Nhân Lỗi Gemma 4 31B: Systematic Bias

Gemma có **một vấn đề gốc rễ duy nhất** biểu hiện qua nhiều dạng:

### Vấn đề gốc: "Step 3 Collapse" — Không Phân Biệt Được Mục Đích Đưa Bằng Chứng

Gemma **luôn kết luận Step 3 = Yes** (offered for truth) khi thấy bất kỳ connection nào giữa statement và fact cần chứng minh, bất kể connection đó là:

| Connection thực tế | Nhãn đúng | Gemma phán đoán |
|--|:--:|:--:|
| Offered to prove **listener's knowledge** | Not Hearsay | Hearsay ❌ |
| Offered to prove **effect on listener** | Not Hearsay | Hearsay ❌ |
| Offered to prove **declarant's state of mind** (direct) | Not Hearsay | Hearsay ❌ |
| Offered to prove **conduct/action** (non-assertive) | Not Hearsay | Hearsay ❌ |
| Offered to prove **truth of content** | Hearsay | Hearsay ✅ |

### Pattern cụ thể trong reasoning Gemma

```
Mẫu điển hình — idx=47:
"On the issue of whether Jane KNEW that Arthur was having an affair,
 the fact that Bob told Jane he suspected Arthur was up to something."

Gemma Step3: "Yes, the evidence is being introduced to establish that 
  Jane had KNOWLEDGE of Arthur's possible infidelity. To prove Jane 
  'knew' or was alerted to the affair, the court must rely on the TRUTH 
  of Bob's assertion that Arthur was 'up to something'."

Lý luận sai: Gemma nhầm "cần biết Bob nói gì" với "cần tin Bob nói đúng"
Đúng ra: Chỉ cần chứng minh Jane NGHE thấy statement → Effect on listener
         Không cần statement của Bob phải TRUE.
```

### Ba Dạng Biểu Hiện

**Dạng 1 — Purpose Confusion (12/17 mẫu):**
Nhầm "offered to prove awareness/knowledge" thành "offered to prove truth"

**Dạng 2 — Conduct Misclassification (3/17 mẫu):**
Nhầm hành vi không có chủ đích communicate (non-assertive conduct) thành statement:
- Poster lyrics → "written assertion"
- Chẩn đoán y khoa → "oral assertion"
- Hồ sơ thi bằng lái → "written assertion"

**Dạng 3 — In-court Confusion (1/17 mẫu):**
Không nhận ra testimony in-court không phải hearsay dù kể lại statement cũ

--

## 5. Tại Sao DeepSeek Làm Tốt Hơn?

### 5.1 Không Có Bias Hệ Thống

DeepSeek có 5 FP và 5 FN — phân bổ hoàn toàn cân bằng. Điều này cho thấy:
- Không học được shortcut "mọi thứ là Hearsay" như Gemma
- Không bị ảnh hưởng bởi JSON parsing như GPT

### 5.2 Lỗi Của DeepSeek: Genuine Hard Cases

10 lỗi của DeepSeek đều là những mẫu **khó thực sự**, được phân bổ đều:

| idx | GT | Pred | Slice | Phân tích |
|:--:|:--:|:--:|--|--|
| 12 | False | True | Non-assertive conduct | Chẩn đoán y khoa (khó) |
| 23 | False | True | Statement made in-court | In-court testimony (khó) |
| 39 | True | False | Standard hearsay | Nonassertive conduct ngược |
| 42 | True | False | Standard hearsay | Reasoning quá phức tạp |
| 68 | True | False | Non-verbal hearsay | Conduct vs statement |
| 72 | False | True | Not introduced to prove truth | Protest signs (khó) |
| 75 | True | False | Standard hearsay | Edge case |
| 77 | False | True | Not introduced to prove truth | Effect on listener |
| 79 | False | True | Not introduced to prove truth | State of mind |
| 84 | True | False | Standard hearsay | Embedded hearsay |

### 5.3 Overlap Phân Tích: Mẫu "Khó Phổ Quát"

Chỉ có **2 mẫu** cả 3 mô hình đều sai:

| idx | Slice | Vì sao khó |
|:--:|--|--|
| **23** | Statement made in-court | Daniel làm chứng về statement của chính mình với vợ → ranh giới in-court/out-of-court mờ nhạt |
| **72** | Not introduced to prove truth | Protest signs — hành vi tập thể, có cả assertive conduct và effect on audience |

--

## 6. Tổng Hợp Nguyên Nhân

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                          │
├──────────────┬──────────────────────────────────────────────────┤
│ DeepSeek     │ Genuine reasoning errors on hard cases           │
│ (10 errors)  │ Balanced FP/FN → no systematic bias              │
│              │ No infrastructure issues                          │
├──────────────┼──────────────────────────────────────────────────┤
│ GPT OSS 20B  │ [23.5%] JSON parsing failures → false FN         │
│ (17 errors)  │ [76.5%] Over-application of "not-for-truth"      │
│              │         exception to Standard hearsay cases       │
│              │ Conservative bias: FN >> FP                       │
├──────────────┼──────────────────────────────────────────────────┤
│ Gemma 4 31B  │ Step 3 Collapse: cannot distinguish              │
│ (17 errors)  │   "offered to prove truth" vs                    │
│              │   "offered to prove knowledge/effect"             │
│              │ Aggressive bias: FP=17, FN=0                     │
│              │ Perfect on Standard hearsay (simple cases)        │
│              │ Catastrophic on Not-introduced-to-prove-truth     │
└──────────────┴──────────────────────────────────────────────────┘
```

--

## 7. Implication Cho Paper PYTHEN

> [!NOTE]
> **Insight quan trọng:** Hai mô hình (GPT và Gemma) đang sai theo **hướng đối lập nhau** — GPT quá conservative (FN bias), Gemma quá aggressive (FP bias). DeepSeek là mô hình duy nhất không có bias hệ thống.

| Implication | Chi tiết |
|--|--|
| **Gemma F1 tụt do FP bias** | Precision sụt giảm mạnh do 17 FP, dù Recall = 100% |
| **GPT F1 tụt do FN bias + infrastructure** | Recall thấp do 14 FN, trong đó 4 cái có thể fix được |
| **PYTHEN pipeline cần fix JSON robustness** | 4 lỗi GPT có thể được giải quyết bằng retry/fallback |
| **Gemma cần prompt engineering cho Step 3** | Cần làm rõ phân biệt "truth of matter asserted" vs "effect on listener" |
| **Slice "Not introduced to prove truth" là bottleneck** | Khó nhất cho Gemma (35% acc), cần special handling |

--

*Root cause analysis — PYTHEN Paper Experiments — 20/08/2026*
