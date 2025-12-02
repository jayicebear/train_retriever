from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
from pathlib import Path
import wandb

wandb.init(project="rag-embedding-training", name="qwen3-finetuning")

def load_train_data(file_path):
    """JSON 파일에서 학습 데이터 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"학습 데이터: {len(data)}개 로드 완료")
    return data


def create_train_examples(train_data):
    """데이터를 InputExample로 변환"""
    train_examples = [
        InputExample(texts=[item['question'], item['content']])
        for item in train_data
    ]
    print(f"InputExample {len(train_examples)}개 생성 완료")
    return train_examples


def calculate_warmup_steps(num_examples, batch_size, epochs, warmup_ratio=0.1):
    """Warmup steps 자동 계산"""
    steps_per_epoch = num_examples // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(int(total_steps * warmup_ratio), 1)
    
    # Warmup이 너무 길면 제한
    if warmup_steps > total_steps * 0.5:
        warmup_steps = int(total_steps * 0.1)
    
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    return warmup_steps


def train_model(model_name, train_examples, batch_size=16, epochs=3, warmup_steps=None):
    """모델 학습"""
    # 모델 로드
    model = SentenceTransformer(model_name)
    
    # Warmup steps 자동 계산
    if warmup_steps is None:
        warmup_steps = calculate_warmup_steps(len(train_examples), batch_size, epochs)
    
    # DataLoader 생성
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Loss 함수
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 학습
    print(f"\n학습 시작... (Epochs: {epochs}, Batch size: {batch_size})")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        callback=wandb.log
    )
    wandb.finish()
    return model


def save_model(model, output_path):
    """학습된 모델 저장"""
    # 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.save(output_path)
    print(f"\n모델 저장 완료: {output_path}")


def main(file_name, model_name="Qwen/Qwen3-Embedding-0.6B", batch_size=16, epochs=3):
    """전체 학습 파이프라인"""
    # 경로 설정
    train_data_path = f"/home/ljm/chunking/train_test_data/{file_name}_train_data.json"
    output_path = f"./trained_model/qwen3_finetuned_retriever_{file_name}"
    
    print(f"=== 모델 학습 시작: {file_name} ===\n")
    
    # 1. 데이터 로드
    train_data = load_train_data(train_data_path)
    
    # 2. InputExample 생성
    train_examples = create_train_examples(train_data)
    
    # 3. 모델 학습
    model = train_model(
        model_name=model_name,
        train_examples=train_examples,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # 4. 모델 저장
    save_model(model, output_path)
    
    print(f"\n=== 학습 완료 ===")


if __name__ == "__main__":
    file_name = 'hallucination_last'
    main(file_name, batch_size=4, epochs=2) 
