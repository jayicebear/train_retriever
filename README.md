# Qwen3-Embedding-0.6B 기반 Retriever 파인튜닝 코드

이 레포지토리는 **Qwen3-Embedding-0.6B** 모델을 Sentence-Transformers 라이브러리로 **Multiple Negatives Ranking Loss** 를 사용해 파인튜닝하여 고성능 한국어 Retriever(검색용 임베딩 모델)를 만드는 코드입니다.

RAG(Retrieval-Augmented Generation) 시스템에서 **검색 정확도**를 극대적으로 끌어올리고 싶을 때 사용하는 파인튜닝 스크립트입니다.

## 왜 이 방식으로 파인튜닝하나요?

### 1. Multiple Negatives Ranking Loss (MNR Loss)
- 한 배치 안에 `(question, positive passage)` 쌍만 넣어줍니다.
- 모델은 **in-batch negative** 방식을 사용해 같은 배치 내 다른 passage들을 자동으로 negative sample로 간주합니다.
- 즉, 별도로 hard negative를 만들지 않아도 매우 효과적인 contrastive learning이 가능합니다.
