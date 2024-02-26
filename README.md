# Level2-nlp-mrc-nlp-06: MRC (Machine Reading Comprehension)

## 📌 대회 설명
## MRC (2024.02.05 ~ 2024.02.23)
<aside>
💡 ODQA (Open-Domain Question Answering)

**주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾은 후 그에 기반하여 다양한 질문에 대한 대답을 추출 or 생성**

- 프로젝트 기간 (2024.02.05 ~ 2024.02.23)
- 학습/평가 데이터
    - `train_dataset` (`datasets.Dataset`)
      - `train` : 3952 샘플
      - `validation` : 240 샘플
    - `test_dataset` 
      - 전체 600 샘플 (Public 240 샘플, Private 360 샘플)
    - Retrieval 문서 집합 `corpus`
      - 60613 개의 중복된 문서 리스트 (56737 개의 독립된 문)

- 평가 지표 : KLUE-RE evaluation metric을 그대로 재현
    1. **Exact Match (EM)** : 모델의 예측과, 실제 답이 정확하게 일치

    2. **F1 Score** : 겹치는 단어에 대한 부분점수 부여



## 📌 팀 소개

* **Team명** : 찐친이 되어줘 [NLP 6조]

|                            김재현                            |                            서동해                            |                            송민환                            |                            장수정                            |                            황예원                            |                            황재훈                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![재현](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/fa007f29-007b-42c0-bb1a-f95176ad7d93) | ![동해-PhotoRoom png-PhotoRoom](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/7ba86ba4-cd7a-4366-97aa-7669e7994a78) | ![민환](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/a3614eb6-4757-4390-9196-f82a455b4418) | ![수정](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/39b8b55c-d1d8-4125-bbf2-11a695bcbc23) | ![예원-PhotoRoom png-PhotoRoom](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/46ab92c3-e6cc-455a-b9c3-a225c8730048) | ![재훈-removebg-preview](https://github.com/boostcampaitech6/level2-klue-nlp-06/assets/82081872/5d8cf554-d59a-44fa-802d-38bd66111263) |
|           [Github](https://github.com/finn-sharp)            |           [Github](https://github.com/DonghaeSuh)            |           [Github](https://github.com/codestudy25)           |             [Github](https://github.com/jo9392)              |             [Github](https://github.com/yeowonh)             |           [Github](https://github.com/iloveonsen)            |
|                [Mail](penguin-klg@jnu.ac.kr)                 |                [Mail](donghaesuh2@gmail.com)                 |                [Mail](meenham_song@naver.com)                |                 [Mail](jo23892389@gmail.com)                 |                  [Mail](yeowonh@sju.ac.kr)                   |                  [Mail](mgs05144@gmail.com)                  |



## 📌 실험한 것들

#### Retrieval

1. Spare Passage Retrieval (SPR)
2. BM25 (SPR)
3. Dense Passage Retrieval (DPR)

#### MRC

1. Extractive MRC
2. Generative MRC
3. Combined MRC based on `pytorch-lightning` with `RAG` algorithm

## 📌 코드 구조

```files
┣ 📂.github
┃ ┣ 📂ISSUE_TEMPLATE
┃ ┃ ┣ bug_report.md
┃ ┃ ┗ feature_request.md
┃ ┣ .keep
┃ ┗ PULL_REQUEST_TEMPLATE.md
┣ 📂code
┃ ┣ 📂retrieval
┃ ┃ ┣ retrieval_BM25.py
┃ ┃ ┗ retrieval_TFIDF.py
┃ ┣ train.py
┃ ┣ trainer_qa.py
┃ ┣ utils_qa.py
┃ ┣ retrieval.py
┃ ┣ inference.py
┃ ┣ arguments.py
┃ ┣ train_generative.py
┃ ┣ arguments_generative.py
┃ ┣ squad_generative.py
┃ ┣ inference_bm25.py
┃ ┣ arguments_extractive.py
┃ ┣ retrieval_dpr.py
┃ ┣ inference_dpr.py
┃ ┣ arguments_dpr.py
┃ ┗ evaluate.py
┣ 📂rag
┃ ┣ indexing.py
┃ ┣ rag_token_run_test.ipynb
┃ ┣ rag_token_train.ipynb
┃ ┣ rag_token_variant_train.ipynb
┃ ┗ wiki_token.py
┗ 📂notebook
  ┗ preprocessing.ipynb
```

