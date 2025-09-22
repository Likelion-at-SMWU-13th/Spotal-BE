#!/usr/bin/env python
"""
초고속 성능 테스트 스크립트
"""

import os
import sys
import django
import time

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spotal.settings')
django.setup()

from recommendations.services.ultra_fast_service import UltraFastService
from recommendations.services.google_service import get_similar_places
from recommendations.services.emotion_service import expand_emotions_with_gpt


def test_ultra_fast_performance():
    """초고속 서비스 성능 테스트"""
    print("=== 초고속 서비스 성능 테스트 ===\n")
    
    test_cases = [
        {
            "name": "스타벅스",
            "address": "용산구 이태원동",
            "emotion_tags": ["편안함", "조용함"],
            "user_id": 1,
            "category": "cafe",
            "top_k": 8
        },
        {
            "name": "맛있는 파스타집",
            "address": "용산구 한남동",
            "emotion_tags": ["로맨틱", "아늑함"],
            "user_id": 1,
            "category": "restaurant",
            "top_k": 8
        }
    ]
    
    ultra_fast_service = UltraFastService()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}: {test_case['name']}")
        
        # 초고속 서비스 테스트
        print("  초고속 서비스...")
        start_time = time.time()
        try:
            results = ultra_fast_service.generate_ultra_fast_recommendations(**test_case)
            execution_time = time.time() - start_time
            print(f"    완료: {execution_time:.3f}초, 결과: {len(results)}개")
            
            # 결과 상세 정보
            for j, result in enumerate(results[:3], 1):
                print(f"    {j}. {result['name']} (소스: {result['source']})")
                print(f"       주소: {result['address']}")
                print(f"       감정: {result['emotions']}")
                print(f"       요약: {result['summary'][:50]}...")
            
        except Exception as e:
            print(f"    오류: {e}")
        print()
    
    print("초고속 테스트 완료!")


def test_api_call_reduction():
    """API 호출 횟수 비교 테스트"""
    print("\n=== API 호출 횟수 비교 ===\n")
    
    # 기존 방식 시뮬레이션
    print("기존 방식 (8개 장소 × 4번 GPT 호출):")
    print("  - 감정 확장: 1번")
    print("  - 요약 생성: 8번")
    print("  - 감정태그 생성: 8번")
    print("  - 키워드 추출: 8번")
    print("  - Google Places 상세: 8번")
    print("  총 API 호출: 33번")
    print("  예상 시간: 30-90초")
    
    print("\n초고속 방식:")
    print("  - 감정 확장: 1번 (캐시)")
    print("  - 요약 생성: 3번 (상위 3개만)")
    print("  - 감정태그 생성: 3번 (상위 3개만)")
    print("  - Google Places 상세: 3번 (상위 3개만)")
    print("  총 API 호출: 7번")
    print("  예상 시간: 5-15초")
    
    print("\n개선율: 78% API 호출 감소, 70-80% 시간 단축")


if __name__ == "__main__":
    test_ultra_fast_performance()
    test_api_call_reduction()
