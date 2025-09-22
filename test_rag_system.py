"""
RAG 시스템 테스트 스크립트
사용법: python test_rag_system.py
"""

import os
import sys
import django

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spotal.settings')
django.setup()

from recommendations.services.rag_service import RAGRecommendationService, get_rag_recommendations
from recommendations.services.enhanced_recommendation_service import EnhancedRecommendationService
from recommendations.models import Place, PlaceEmbedding
from django.contrib.auth import get_user_model

User = get_user_model()


def test_rag_search():
    """RAG 검색 테스트"""
    print("=== RAG 검색 테스트 ===")
    
    # 테스트 쿼리들
    test_queries = [
        "용산에서 분위기 좋은 카페",
        "감성적인 맛집",
        "조용한 독서 카페",
        "데이트하기 좋은 레스토랑"
    ]
    
    rag_service = RAGRecommendationService()
    
    for query in test_queries:
        print(f"\n검색 쿼리: '{query}'")
        try:
            results = rag_service.hybrid_search(query, top_k=3)
            if results:
                for i, (place, score) in enumerate(results, 1):
                    print(f"  {i}. {place.name} (점수: {score:.3f})")
                    print(f"     주소: {place.address}")
                    print(f"     감정: {list(place.emotions.values_list('name', flat=True))}")
            else:
                print("  결과 없음")
        except Exception as e:
            print(f"  오류: {e}")


def test_enhanced_recommendation():
    """개선된 추천 시스템 테스트"""
    print("\n=== 개선된 추천 시스템 테스트 ===")
    
    enhanced_service = EnhancedRecommendationService()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "스타벅스",
            "address": "용산구 이태원동",
            "emotion_tags": ["편안함", "조용함"],
            "category": "cafe"
        },
        {
            "name": "맛있는 파스타집",
            "address": "용산구 한남동",
            "emotion_tags": ["로맨틱", "아늑함"],
            "category": "restaurant"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {test_case['name']}")
        try:
            results = enhanced_service.generate_enhanced_recommendations(
                name=test_case["name"],
                address=test_case["address"],
                emotion_tags=test_case["emotion_tags"],
                category=test_case["category"],
                top_k=3
            )
            
            if results:
                for j, result in enumerate(results, 1):
                    place_data = result['place']
                    print(f"  {j}. {place_data['name']} (소스: {result['source']})")
                    print(f"     주소: {place_data['address']}")
                    print(f"     감정: {place_data['emotions']}")
                    if result.get('rag_score'):
                        print(f"     RAG 점수: {result['rag_score']:.3f}")
            else:
                print("  결과 없음")
        except Exception as e:
            print(f"  오류: {e}")


def test_embedding_status():
    """임베딩 상태 확인"""
    print("\n=== 임베딩 상태 확인 ===")
    
    total_places = Place.objects.count()
    places_with_embedding = PlaceEmbedding.objects.count()
    
    print(f"전체 장소 수: {total_places}")
    print(f"임베딩이 있는 장소 수: {places_with_embedding}")
    print(f"임베딩 비율: {places_with_embedding/total_places*100:.1f}%" if total_places > 0 else "장소 없음")
    
    if places_with_embedding > 0:
        print("\n임베딩이 있는 장소들:")
        for embedding in PlaceEmbedding.objects.select_related('place')[:5]:
            place = embedding.place
            print(f"  - {place.name} (모델: {embedding.model_name})")


def test_user_personalization():
    """사용자 개인화 테스트"""
    print("\n=== 사용자 개인화 테스트 ===")
    
    # 첫 번째 사용자 찾기
    user = User.objects.first()
    if not user:
        print("사용자가 없습니다.")
        return
    
    print(f"테스트 사용자: {user.username or user.email or user.id}")
    
    # 개인화된 추천
    try:
        from recommendations.services.rag_service import get_personalized_recommendations
        recommendations = get_personalized_recommendations(user, top_k=3)
        
        if recommendations:
            print("개인화된 추천:")
            for i, place in enumerate(recommendations, 1):
                print(f"  {i}. {place.name}")
                print(f"     주소: {place.address}")
                print(f"     감정: {list(place.emotions.values_list('name', flat=True))}")
        else:
            print("개인화된 추천 결과 없음")
    except Exception as e:
        print(f"개인화 추천 오류: {e}")


def main():
    """메인 테스트 함수"""
    print("RAG 시스템 테스트 시작...\n")
    
    # 1. 임베딩 상태 확인
    test_embedding_status()
    
    # 2. RAG 검색 테스트
    test_rag_search()
    
    # 3. 개선된 추천 시스템 테스트
    test_enhanced_recommendation()
    
    # 4. 사용자 개인화 테스트
    test_user_personalization()
    
    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    main()
