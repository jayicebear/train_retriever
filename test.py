from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


def evaluate_retrieval_model(
    model_path,
    test_data_path,
    k_values=[1, 3, 5, 10],
    num_samples=3
):
    """임베딩 모델 평가 함수"""
    
    print(f"\n{'='*60}")
    print(f"모델 평가: {model_path}")
    print(f"{'='*60}\n")
    
    # 1. 모델 및 데이터 로드
    model = SentenceTransformer(model_path)
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = [item['question'] for item in data]
    chunks = [item['content'] for item in data]
    print(f"✓ 테스트 데이터: {len(data)}개\n")
    
    # 2. 임베딩 생성
    print("임베딩 생성 중...")
    q_emb = model.encode(questions, batch_size=32, show_progress_bar=True)
    c_emb = model.encode(chunks, batch_size=32, show_progress_bar=True)
    
    
    # 3. 평가
    print(f"\n{'='*60}")
    print("평가 결과")
    print(f"{'='*60}\n")
    
    results = {}
    for k in k_values:
        correct = sum(
            i in np.argsort(cosine_similarity([q_emb[i]], c_emb)[0])[::-1][:k]
            for i in range(len(questions))
        )
        accuracy = correct / len(questions) * 100
        results[k] = accuracy
        print(f"Top-{k:2d} Accuracy: {accuracy:6.2f}% ({correct}/{len(questions)})")
    
    # 4. 샘플 출력
    print(f"\n{'='*60}")
    print(f"샘플 결과 (처음 {num_samples}개)")
    print(f"{'='*60}\n")
    
    for i in range(min(num_samples, len(questions))):
        sims = cosine_similarity([q_emb[i]], c_emb)[0]
        top_idx = np.argsort(sims)[::-1][:max(k_values)]
        rank = np.where(top_idx == i)[0]
        
        print(f"[질문 {i+1}]: {questions[i][:80]}...")
        print(f"정답 순위: {rank[0]+1 if len(rank) > 0 else 'Not in top-k'}\n")
        
        for r, idx in enumerate(top_idx[:3], 1):
            marker = "✓" if idx == i else " "
            print(f"{marker} {r}위 ({sims[idx]:.4f}): {chunks[idx][:60]}...")
        print()
    
    return results


if __name__ == "__main__":
    # 사용 예시
    file_name = 'hallucination_last'
    
    results = evaluate_retrieval_model(
        model_path=f"./trained_model/qwen3_finetuned_retriever_{file_name}",
        test_data_path=f"/home/ljm/chunking/train_test_data/{file_name}_test_data.json",
        k_values=[1, 3, 5, 10],
        num_samples=3
    )
