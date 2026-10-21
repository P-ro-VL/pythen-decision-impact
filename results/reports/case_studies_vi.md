# Báo cáo Phân tích Lỗi và Case Study (Error Analysis & Case Studies)
**Bộ dữ liệu:** Hearsay (94 mẫu) | **Mô hình đối chiếu:** Gemma 4B, DeepSeek V3.2, GPT OSS 20B

--

## 1. Bảng Thống Kê Phân Bố Lỗi (Error Distribution Table)

Dưới đây là thống kê chi tiết số lượng và tỷ lệ phần trăm (%) xuất hiện các mã lỗi từ **E1 đến E5** trên từng mô hình. 

> [!NOTE]
> - Tỷ lệ phần trăm trong bảng được tính trên tổng số mẫu của toàn bộ tập dữ liệu (**94 mẫu**) để phản ánh trực quan xác suất xảy ra lỗi của mô hình trên thực tế.
> - Một mẫu có thể chứa nhiều lỗi trích xuất song song (ví dụ: vừa sai E1 trên predicate này vừa sai E2 trên predicate khác), nên tổng tỷ lệ của một dòng có thể vượt quá tỷ lệ mẫu sai thực tế.

### Bảng 1: Tỷ lệ xuất hiện lỗi trên tổng số mẫu đánh giá (94 mẫu)

| Mô hình | E1 (Hallucination - FP)<br>Tầng Extraction | E2 (Omission - FN)<br>Tầng Extraction | E3 (No Reference)<br>Tầng Extraction | E4 (Over-conservatism)<br>Tầng CoT | E5 (Logical Jump)<br>Tầng CoT |
|--|:--:|:--:|:--:|:--:|:--:|
| **Gemma 4B** | **23.40%** (22 mẫu) | **15.96%** (15 mẫu) | 0.00% (0 mẫu) | 0.00% (0 mẫu) | 0.00% (0 mẫu) |
| **DeepSeek V3.2** | **41.49%** (39 mẫu) | **24.47%** (23 mẫu) | **9.57%** (9 mẫu) | **4.26%** (4 mẫu) | **1.06%** (1 mẫu) |
| **GPT OSS 20B** | **37.23%** (35 mẫu) | **18.09%** (17 mẫu) | 0.00% (0 mẫu) | **14.89%** (14 mẫu) | 0.00% (0 mẫu) |

### Bảng 2: Tỷ lệ đóng góp của từng mã lỗi trong tổng số lỗi ghi nhận của mô hình

Bảng này giúp hiểu rõ phân phối nội bộ các loại lỗi của từng mô hình (tổng cộng 100% cho mỗi mô hình dựa trên tổng số lượng lỗi E1–E5 xuất hiện).

| Mô hình | Tổng số lỗi | E1 (%) | E2 (%) | E3 (%) | E4 (%) | E5 (%) |
|--|:--:|:--:|:--:|:--:|:--:|:--:|
| **Gemma 4B** | 37 | **59.46%** | **40.54%** | 0.00% | 0.00% | 0.00% |
| **DeepSeek V3.2** | 76 | **51.32%** | **30.26%** | **11.84%** | **5.26%** | **1.32%** |
| **GPT OSS 20B** | 66 | **53.03%** | **25.76%** | 0.00% | **21.21%** | 0.00% |

> [!IMPORTANT]
> **Nhận xét chính:**
> - **E1 (Predicate Hallucination)** là lỗi phổ biến nhất ở cả 3 mô hình, đặc biệt nghiêm trọng ở DeepSeek V3.2 (41.49% số mẫu bị gán nhãn True sai).
> - **E4 (Over-conservatism Bias)** chỉ xuất hiện ở tầng suy luận CoT (nơi mô hình tự do lập luận). GPT OSS 20B bị ảnh hưởng nặng nề nhất bởi lỗi này (14.89% số mẫu), nơi nó lý luận đúng ở các bước đầu nhưng lại kết luận phủ nhận một cách phòng thủ ở bước cuối.
> - **Gemma 4B** hoàn toàn không có lỗi tầng CoT trong phân tích này do lỗi của Gemma tập trung 100% vào tầng trích xuất logic sơ bộ (E1 & E2).

--

## 2. Case Study Tiêu Biểu: Trích xuất sai Vị tự Ranh giới (Boundary Predicate)

Dưới đây là 2 case study điển hình minh họa trường hợp Gemma trích xuất sai vị tự ranh giới (Boundary Predicate) dẫn đến hệ thống Neuro-Symbolic (PYTHEN) suy luận sai, trong khi DeepSeek trích xuất chính xác.

### Case Study 1: Mẫu số 6 (Slice: Non-assertive conduct)

* **Tình huống pháp lý (Fact Pattern):**
  > *"On the issue of whether Patricia was a fan of Coldplay, the fact that she had a poster with the lyrics of 'Viva la Vida' on her bedroom wall."*
  *(Để xác định liệu Patricia có phải là người hâm mộ Coldplay hay không, việc cô ấy treo một bức poster có lời bài hát "Viva la Vida" trên tường phòng ngủ của mình.)*
* **Ground Truth (Nhãn chuẩn):** **`False`** (Không phải Hearsay / Not Hearsay)
* **Phân tích bản chất pháp lý:**
  Bức poster chứa lời bài hát là một văn bản ngoài tòa (out-of-court written statement). Tuy nhiên, nó được đưa ra làm bằng chứng nhằm chứng minh Patricia thích ban nhạc Coldplay (thể hiện qua hành vi treo poster của cô ấy). Mục đích ở đây **không nhằm chứng minh tính chân thực của lời bài hát** (ví dụ: chứng minh các tuyên bố trong bài "Viva la Vida" là đúng sự thật khách quan).
  Vì vậy, vị tự ranh giới cốt lõi quyết định ranh giới Hearsay ở đây là **`is_introduced_to_prove_truth`** phải là **`False`**.

```mermaid
graph TD
    A[Bằng chứng: Treo poster Coldplay] -> B{Mục đích là gì?}
    B ->|Để chứng minh cô ấy là fan| C[Gián tiếp chỉ ra hành vi/sở thích]
    B ->|Để chứng minh lời bài hát là thật| D[Chứng minh nội dung phát ngôn]
    C -> E[is_introduced_to_prove_truth = False]
    D -> F[is_introduced_to_prove_truth = True]
    E -> G[NOT HEARSAY]
    F -> H[HEARSAY]
    style E fill:#d4edda,stroke:#28a745,stroke-width:2px
    style G fill:#d4edda,stroke:#28a745,stroke-width:2px
```

* **So sánh trích xuất giữa hai mô hình:**

| Tham số / Vị tự | Ground Truth | Trích xuất của Gemma 4B | Trích xuất của DeepSeek V3.2 |
|--|:--:|:--:|:--:|
| `is_oral_assertion` | False | False | False |
| `is_written_assertion` | True | True | True |
| `is_non_verbal_assertive_conduct` | False | False | False |
| `is_out_of_court` | True | True | True |
| **`is_introduced_to_prove_truth`** (Ranh giới) | **False** | <span style="color:red">**True** (Sai - E1)</span> | <span style="color:green">**False** (Đúng)</span> |
| **PYTHEN Inference** | **False (Not Hearsay)** | <span style="color:red">**True (Hearsay)**</span> | <span style="color:green">**False (Not Hearsay)**</span> |

* **Hệ quả suy luận:**
  - **Gemma 4B** mắc lỗi **E1 (Predicate Hallucination)** khi gán `is_introduced_to_prove_truth: True`. Do đó, PYTHEN áp dụng công thức logic AND và đưa ra kết quả sai lệch: `(False ∨ True ∨ False) ∧ True ∧ True = True` (False Positive).
  - **DeepSeek V3.2** trích xuất chính xác vị tự ranh giới này là `False`, giúp PYTHEN suy luận đúng: `(False ∨ True ∨ False) ∧ True ∧ False = False`.

--

### Case Study 2: Mẫu số 9 (Slice: Non-assertive conduct)

* **Tình huống pháp lý (Fact Pattern):**
  > *"On the issue of whether Richard had ever visited Chicago, the fact that he gave a speech there in 2005."*
  *(Để xác định liệu Richard đã từng đến Chicago hay chưa, việc anh ấy có một bài phát biểu ở đó vào năm 2005.)*
* **Ground Truth (Nhãn chuẩn):** **`False`** (Không phải Hearsay / Not Hearsay)
* **Phân tích bản chất pháp lý:**
  Việc Richard phát biểu tại Chicago là một hành động vật lý (physical conduct/presence). Hành động này được đưa ra để chứng minh sự hiện diện của Richard tại Chicago vào năm 2005 (anh ấy ở đó thì mới phát biểu được). Ở đây, **không có phát ngôn hearsay nào được sử dụng để chứng minh sự thật** (bản thân bài phát biểu nói gì không quan trọng, quan trọng là hành động phát biểu xảy ra tại địa điểm đó).
  Do đó, vị tự ranh giới cốt lõi ở đây là các vị tự định danh phát ngôn (**`is_oral_assertion`**, `is_written_assertion`, `is_non_verbal_assertive_conduct`) phải là **`False`**.

```mermaid
graph TD
    A[Bằng chứng: Richard phát biểu tại Chicago] -> B{Mục đích là gì?}
    B ->|Chứng minh Richard có mặt ở đó| C[Hành vi vật lý / Physical presence]
    B ->|Chứng minh những gì Richard nói là đúng| D[Phát ngôn hearsay]
    C -> E[is_oral_assertion = False]
    D -> F[is_oral_assertion = True]
    E -> G[NOT HEARSAY]
    F -> H[HEARSAY]
    style E fill:#d4edda,stroke:#28a745,stroke-width:2px
    style G fill:#d4edda,stroke:#28a745,stroke-width:2px
```

* **So sánh trích xuất giữa hai mô hình:**

| Tham số / Vị tự | Ground Truth | Trích xuất của Gemma 4B | Trích xuất của DeepSeek V3.2 |
|--|:--:|:--:|:--:|
| **`is_oral_assertion`** (Ranh giới) | **False** | <span style="color:red">**True** (Sai - E1)</span> | <span style="color:green">**False** (Đúng)</span> |
| `is_written_assertion` | False | False | False |
| `is_non_verbal_assertive_conduct` | False | False | False |
| `is_out_of_court` | True | True | True |
| `is_introduced_to_prove_truth` | True | <span style="color:red">**True** (Sai)</span> | **True** (Sai nhưng không ảnh hưởng) |
| **PYTHEN Inference** | **False (Not Hearsay)** | <span style="color:red">**True (Hearsay)**</span> | <span style="color:green">**False (Not Hearsay)**</span> |

* **Hệ quả suy luận:**
  - **Gemma 4B** gán nhầm hành động phát biểu là một phát ngôn hearsay khẩu ngữ (`is_oral_assertion: True`), dẫn đến PYTHEN suy luận sai: `(True ∨ False ∨ False) ∧ True ∧ True = True` (False Positive).
  - **DeepSeek V3.2** nhận diện chính xác đây chỉ là hành vi vật lý thể hiện sự hiện diện trực tiếp, gán `is_oral_assertion: False`. Nhờ vậy, dù DeepSeek có gán nhầm vị tự mục đích là True thì biểu thức logic của PYTHEN vẫn triệt tiêu lỗi: `(False ∨ False ∨ False) ∧ True ∧ True = False` (Kết quả Đúng).
