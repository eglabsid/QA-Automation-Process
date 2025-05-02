import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

def calculate_embedding_distances(categories_with_items):
    """
    각 상위 항목과 그에 속한 하위 항목 간의 임베딩 거리를 계산합니다.
    
    Args:
        categories_with_items: 상위 항목과 하위 항목을 포함하는 딕셔너리
            예: {'Rendering': ['clipping', 'Zfighting', 'LOD', 'tearing', 'Shadow'], ...}
    
    Returns:
        각 하위 항목과 해당 상위 항목 간의 거리를 정렬한 결과
    """
    # 1. 문장 임베딩 모델 로드
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # 다국어 지원 모델
    
    results = []
    
    # 각 카테고리별로 처리
    for category, items in categories_with_items.items():
        # 카테고리와 아이템의 임베딩 생성
        category_embedding = model.encode([category])[0].reshape(1, -1)
        item_embeddings = model.encode(items)
        
        # 각 아이템과 카테고리 간의 코사인 유사도 계산
        category_results = []
        
        for i, item in enumerate(items):
            item_embedding = item_embeddings[i].reshape(1, -1)
            # 유사도 계산 (코사인 유사도는 값이 클수록 유사함을 의미)
            similarity = cosine_similarity(item_embedding, category_embedding)[0][0]
            distance = 1 - similarity  # 유사도를 거리로 변환
            
            category_results.append({
                "category": category,
                "item": item,
                "similarity": similarity,
                "distance": distance
            })
        
        # 거리에 따라 정렬
        category_results.sort(key=lambda x: x["distance"])
        results.append(category_results)
    
    return results

def print_results(results):
    """결과를 보기 좋게 출력합니다."""
    for category_results in results:
        if not category_results:  # 결과가 비어있으면 건너뜀
            continue
            
        category = category_results[0]["category"]
        print(f"\n카테고리: {category}")
        print("-" * 50)
        print(f"{'항목':<15} {'거리':<10} {'유사도':<10}")
        print("-" * 50)
        
        for result in category_results:
            print(f"{result['item']:<15} {result['distance']:.4f} {result['similarity']:.4f}")

def main():
    # 상위 항목과 하위 항목 정의
    categories_with_items = {}
    
    while True:
        print("\n상위 항목을 입력하세요 (종료하려면 빈 칸 입력):")
        category = input().strip()
        if not category:
            break
            
        print(f"\n'{category}'의 하위 항목들을 입력하세요 (쉼표로 구분):")
        items_input = input().strip()
        items = [item.strip() for item in items_input.split(',')]
        
        categories_with_items[category] = items
    
    if not categories_with_items:
        print("입력된 데이터가 없습니다.")
        return
    
    print("\n임베딩 거리 계산 중...")
    results = calculate_embedding_distances(categories_with_items)
    print_results(results)
    
    # 전체 결과를 DataFrame으로 변환하여 CSV로 저장 (선택적)
    save_csv = input("\n결과를 CSV 파일로 저장하시겠습니까? (y/n): ")
    if save_csv.lower() == 'y':
        all_rows = []
        for category_results in results:
            for result in category_results:
                all_rows.append(result)
        
        df = pd.DataFrame(all_rows)
        filename = "embedding_distances.csv"
        df.to_csv(filename, index=False)
        print(f"결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    main()