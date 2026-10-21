# MÔ TẢ CHUNG
Một hệ thống Neuro-symbolic Legal AI để hỗ trợ việc reasoning, giải thích (explaination) và xác minh (verification).

# BÀI TOÁN DÙNG TRONG EXPERIMENT
Đưa ra quyết đinh xem một fact pattern cho trước có phải là hearsay hay không.

Sử dụng subset `hearsay` của LegalBench để thực hiện đánh giá.

# KIẾN TRÚC
Gồm hai phần:
1. LLM: LLM đóng vai trò là người trích xuất các predicate từ fact pattern cho trước. Các predicate này được dùng làm đầu vào cho phần thứ hai
2. Symbolic System: Hệ thống sử dụng PYTHEN framework (https://arxiv.org/pdf/2603.15317) làm symbolic system để thực hiện suy luận. Mô tả ngắn gọn về PYTHEN: Định nghĩa trước một rule tree dựa trên bộ câu hỏi xác định xem 1 fact pattern có phải là hearsay hay không của LegalBench. Sau đó, sử dụng các predicate trích xuất được làm input để đưa vào rule tree này để đưa ra quyết định. PYTHEN có thể in ra quá trình nó xử lý các predicate để suy luận.

# MỤC TIÊU CỦA PAPER
Chứng minh PYTHEN có thể đóng vai trò hỗ trợ trong việc giải thích và xác minh LLMs.

# KẾT QUẢ EXPERIMENT ĐẠT ĐƯỢC ĐẾN THỜI ĐIỂM HIỆN TẠI
Approach	Model	Accuracy	Precision	Recall	F1-Score
Naive LLMs
DeepSeek V3.2	0.691489	0.603448	0.853659	0.707071
GPT OSS 20B	0.648936	0.666667	0.390244	0.492308
Gemma 4 31B	0.734043	0.673913	0.756098	0.712644
LLMs with Chain-of-Thought
DeepSeek V3.2	0.744681	0.634921	0.97561	0.769231
GPT OSS 20B	0.765957	0.74359	0.707317	0.725
Gemma 4 31B	0.776596	0.666667	0.97561	0.792079
Neuro-symbolic (LLMs with PYTHEN) (Ours)
DeepSeek V3.2	0.776596	0.678571	0.926829	0.783505
GPT OSS 20B	0.819149	0.74	0.902439	0.813187
Gemma 4 31B	0.840426	0.75	0.95122	0.83871

The new result shows that, our approach critically enhances the performance among all models in comparison to naive LLMs call and LLMs with CoT approach. Specifically, F1-Score increases on all models (from 0.769 to 0.784 with DeepSeek, 0.725 to 0.813 with GPT OSS, and from 0.792 to 0.839 with Gemma 4). This results emphasizes on the role of PYTHEN framework, separating legal reasoning component to rule tree in other words, in reducing the conclusion error of LLMs.

In comparison to Naive Approach, PYTHEN indicates a huge improvement across all metrics. For example, GPT OSS 20B's Recall increases dramatically from 0.39 to 0.902. This increase proves that when requesting the model to make the final decision, GPT OSS 20B usually produces False Negative outputs. However, using LLMs to extract predicates and then use PYTHEN as a symbolic resolver significantly reduces False Negatives.

LLMs with CoT have a better result compared to Naive Approach. DeepSeek V3.2 and Gemma 4B have high Recall (both of them are 0.976), but observe low Precision (0.635 and 0.679 respectively). This situation demonstrates the bias in the labeling process: they usually assign `True` predictions to the given fact patterns. By using PYTHEN framework, we resolved this problem: those two models' Precision in our approach has increase to 0.679 and 0.750 respectively (GPT OSS 20B's Precision in two approaches are nearly the same, 0.744 and 0.740). This also evidences that symbolic system like PYTHEN can minimize the False Negative labeling.

Gemma 4B got the highest F1-Score (0.839) and Accuracy (0.840), followed by GPT OSS 20B (0.813) and DeepSeek (0.784). However, the discrepancy between F1-Score of all models is much smaller than those of Naive Approach, which means that  Neuro-symbolic architecture helps stabilize performance across LLM models. 

Overall, our approach contributes to prove the supporting role of PYTHEN framework by formalizing the legal reasoning process with LLMs extracted information. Like other neuro-symbolic systems, LLMs are still responsible for the input quality of the symbolic system by generating appropriate predicates, but PYTHEN framework helps reducing the instability by transforming the problem into a verifiable reasoning process.

DEEPER ANALYSIS ON THE EXPERIMENT RESULT: LOOKING AT ERRORS DISTRIBUTION
I have designed an Error Taxonomy to get in-depth analysis on the models' errors in two important stages of LLMs: Predicate extraction (in Our Approach) and thinking process (in LLMs with CoT Approach). The taxonomy can be described as:
Stage	Code	Name	Description
LLM Extraction (in Neuro-symbolic Approach)
E1	Predicate Hallucination (False Positive)	Extracted JSON sets a predicate to true while the ground-truth predicate table marks it false.
E2	Predicate Omission (False Negative)	Extracted JSON sets a predicate to false while the ground-truth predicate table marks it true.
E3	Invalid Reference	Extracted JSON sets a predicate to true, but reference texts is absent, non-supportive, or not found in the fact pattern.
Thinking Process (LLMs with CoT Approach)
E4	Conservatism Bias	Baseline predicts false for a true ground-truth sample (this captures false-negative CoT conservatism)
E5	Logical Bypass in CoT	Baseline predicts true for a false ground-truth sample (this captures free-form reasoning that jumps over at least one legal condition)

I manually labeled the `hearsay` subset of LegalBench with 5 predicates:
1. `is_oral_assertion`: Does the evidence involve a spoken statement or oral assertion?
2. `is_written_assertion`: Does the evidence involve a written statement or document?
3. `is_non_verbal_assertive_conduct`: Does the evidence involve non-verbal conduct intended as an assertion (e.g., pointing a finger to identify someone)?
4. `is_out_of_court`: Was the statement/conduct made outside of the current trial or hearing?
5. `is_introduced_to_prove_truth`: Is the evidence being introduced to prove the exact truth of what it asserts?

Here is the error distribution table analyzed based on the above taxonomy:
Error	DeepSeek V3.2	GPT OSS 20B	Gemma 4 31B
E1	18.10%	13.80%	13.80%
E2	4.30%	4.30%	2.10%
E3	0.00%	0.00%	0.00%
E4	1.10%	12.80%	1.10%
E5	24.50%	10.60%	21.30%
Total Accuracy:
LLMs with CoT	74.50%	76.60%	77.70%
Neuro-symbolic (Ours)	77.70%	81.90%	84.00%

In LLMs with CoT Approach (E4, E5), error E5 highly comprises 24.5% and 21.3% in DeepSeek and Gemma models respectively, which shows that these two models often "jump" in their thinking process. This behavior leads to "hasty" conclusions while not having enough conditions. Meanwhile, GPT OSS 20B has a high E4 error rate (12.8%), which means GPT OSS often chooses "False" when its conclusion is uncertain. 

In Neuro-symbolic Approach (E1, E2, E3), the most popular error is E1. DeepSeek has the highest proportion (18.1%), while GPT OSS and Gemma are the same (13.8%). This proves that when LLMs extract JSON, their error is mainly assigning `True` to a predicate that is actually not in the given fact pattern (I call this "Predicate Hallucination" as mentioned in the Taxonomy above). E2 is much lower, with 4.30% in both DeepSeek and GPT OSS and 2.10% in Gemma 4, which means those models, especially Gemma 4B, rarely miss predicates. In addition, no model struggles with verbatim referring to the predicates in the given fact pattern (0.0% of E3 in all models). 

Looking at the Total Accuracy scores, LLMs with PYTHEN Approach increase the accuracy across three models than they are in LLMs with CoT approach. PYTHEN Framework undoubtedly helps reduce errors in the conclusion layer, especially when CoT easily gets E4 or E5. However, it is important to note that PYTHEN cannot resolve the predicate extraction error itself, because its inputs depend on LLM's predicate extraction result.

Generally speaking, by breaking down error types LLMs can make in two Approaches, the accuracy score of our approach, especially the PYTHEN framework, strengthen the stability and verification of reasoning process. Though LLM models may still get "Predicate Hallucination" (this can be listed as a Limitation or Discussion), the overall accuracy still increases notably.

ABLATION STUDIES
Oracle Experiment on Gold Predicates and PYTHEN
As mentioned above, I manually labeled 5 predicates of each fact pattern in the `hearsay` subset of LegalBench. To prove that the remaining gap in our Neuro-symbolic Approach comes from predicate extraction, not symbolic inference, I fed the gold predicates directly into the PYTHEN rule tree. 

The Accuracy, Precision, Recall and F1-Score are all 1.0, showing that the current PYTHEN rule tree is logically aligned with the ground truths, which also means the remaining gap in our Neuro-symbolic Approach comes from predicate extraction, not symbolic inference.

Standalone Predicate Extraction Evaluation
The `hearsay` subset has 94 samples, each sample has 5 predicates (mentioned above); therefore, we have 470 predicate slots in total. Here is the performance of LLMs on those predicate slots in Predicate Extraction stage:
Model	Accuracy	Precision	Recall	F1-Score	Macro F1-Score
DeepSeek V3.2	0.845	0.789	0.851	0.819	0.689
GPT OSS 20B	0.843	0.775	0.871	0.82	0.729
Gemma 4 31B	0.911	0.873	0.918	0.894	0.768

Gemma 4 has the best performance across three models in all metrics. Its F1-Score and Accuracy are 0.894 and 0.911 respectively, showing not only its very good prediction but also its good balance between identifying correct predicates and avoiding assigning incorrect predicates. 

GPT OSS 20B and DeepSeek V3.2 have nearly the same F1-Score (0.820 and 0.819 respectively). However, GPT OSS tends to produce more False Positive and less False Negative, while DeepSeek produces less False Positive and more False Negative. DeepSeek is better at Accuracy but lower in Recall in comparison to GPT OSS.

The following table has a deeper look at F1-Score by each predicate:
Predicate	DeepSeek V3.2	GPT OSS 20B	Gemma 4 31B
is_oral_assertion	0.897	0.929	0.957
is_written_assertion	0.286	0.429	0.375
is_non_verbal_assertive_conduct	0.593	0.667	0.667
is_out_of_court	0.91	0.877	0.976
is_introduced_to_prove_truth	0.76	0.744	0.864

It is clear that the predicates are not equally difficult to extract. `is_oral_assertion` and `is_out_of_court` are two predicates that are well extracted (Gemma 4B got 0.957 and 0.976 respectively). 

Meanwhile, `is_written_assertion` is the most difficult predicate to be extracted across three models (0.286, 0.429 and 0.375 with DeepSeek, GPT OSS, Gemma 4 respectively). This evidence shows that models struggle with identifying whether a span in the given fact pattern contains written assertion or it's just an event/evidence having textual elements but not being used as an assertion hearsay. 
For a more closer look, `is_written_assertion` predicate is an extremely difficult predicate to be extracted. It is not as easy as "Are there any written texts in the given fact pattern?", it is "Is there a written assertion made as a statement of affirmation?". LLMs can face one of the following difficulties with this predicate: 
Many samples of the `hearsay` subset contain textual elements but do not constitute hearsay statements.
The boundary between written assertion and non-assertive circumstantial evidence is very thin. For example, the fact pattern "the fact that she had a poster with the lyrics of "Viva la Vida" on her bedroom wall" (this sample's ground truth is "Non-assertive Conduct"), LLMs may notice phrase "the lyrics of" and then instantly assign `is_written_assertion` to True.
Also, in this subset, the number of positive samples of `is_written_assertion` predicate is not large as other predicates. Therefore, even a small amount of False Positive or False Negative error can lead to dramatic decrease in F1-Score. As a consequence, `is_written_assertion` is difficult in both extraction and statistics.

In the Predicate Extraction Evaluation table above, I calculated a metric called "Macro F1-Score". This metric is calculated by averaging F1-Score of each predicate. Unlike F1-Score, which is calculated on all 470 predicate slots, Macro F1-Score considers predicates are equally important. This metric is very useful when some predicates are rarer than others. For example, `is_written_assertion` appears less than other predicates; therefore, normal F1-Score can hide the bad performance of a model on this rare predicate. 
Gemma 4B has 0.894 in F1-Score, but only 0.768 in Macro F1-Score; DeepSeek has 0.819 in F1-Score, but only 0.689 in Macro F1-Score. This indicates that the overall results are dominated by common or easily identifiable predicates, whereas certain rare and challenging predicates, such as `is_written_assertion` or `is_non_verbal_assertive_conduct`, remain weak points. Consequently, relying solely on normal F1-Score or Accuracy may lead to an overly optimistic assessment of the LLM's extraction capabilities.

Generally, this result shows that the remaining error of our approach does not come from PYTHEN framework, but from the Predicate Extraction layer. When predicates are correctly extracted, PYTHEN can perform reliable inference; however, if even a single core predicate is hallucinated or missed, the final decision will be incorrect. Gemma 4B is currently the best extraction model among all models, while a common bottleneck for all of them lies in predicates such as `is_written_assertion` and `is_non_verbal_assertive_conduct`. 

