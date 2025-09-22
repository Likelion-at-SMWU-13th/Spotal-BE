#!/usr/bin/env python
"""
빠른 성능 테스트 스크립트
"""

import os
import sys
import django
import time

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spotal.settings')
django.setup()

from recommendations.services.fast_hybrid_service import FastHybridService
from recommendations.services.google_service import get_similar_places
from recommendations.services.emotion_service import expand_emotions_with_gpt


def test_fast_vs_original():
    """빠른 서비스 vs 기존 서비스 비교"""
    print("=== 빠른 서비스 vs 기존 서비스 비교 ===\n")
    
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
    
    fast_service = FastHybridService()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}: {test_case['name']}")
        
        # 기존 방식 (구글맵만)
        print("  기존 방식 (구글맵만)...")
        start_time = time.time()
        try:
            emotions = expand_emotions_with_gpt(test_case['emotion_tags'])
            emotion_names = [e.name for e in emotions]
            
            if "cafe" in test_case['category'].lower():
                allowed_types = ["cafe"]
            else:
                allowed_types = ["restaurant", "food"]
            
            original_results = get_similar_places(
                test_case['address'], 
                emotion_names, 
                allowed_types=allowed_types
            )[:test_case['top_k']]
            
            original_time = time.time() - start_time
            print(f"    완료: {original_time:.3f}초, 결과: {len(original_results)}개")
        except Exception as e:
            print(f"    오류: {e}")
            original_time = 0
            original_results = []
        
        # 빠른 하이브리드 방식
        print("  빠른 하이브리드 방식...")
        start_time = time.time()
        try:
            fast_results = fast_service.generate_fast_recommendations(**test_case)
            fast_time = time.time() - start_time
            print(f"    완료: {fast_time:.3f}초, 결과: {len(fast_results)}개")
        except Exception as e:
            print(f"    오류: {e}")
            fast_time = 0
            fast_results = []
        
        # 비교
        if original_time > 0 and fast_time > 0:
            if fast_time < original_time:
                improvement = ((original_time - fast_time) / original_time) * 100
                print(f"  개선: {improvement:.1f}% 빨라짐")
            else:
                degradation = ((fast_time - original_time) / original_time) * 100
                print(f"  저하: {degradation:.1f}% 느려짐")
        print()
    
    print("테스트 완료!")


def test_caching_effect():
    """캐싱 효과 테스트"""
    print("\n=== 캐싱 효과 테스트 ===\n")
    
    service = FastHybridService()
    test_case = {
        "name": "스타벅스",
        "address": "용산구 이태원동",
        "emotion_tags": ["편안함", "조용함"],
        "user_id": 1,
        "category": "cafe",
        "top_k": 8
    }
    
    # 첫 번째 실행
    print("첫 번째 실행 (캐시 없음)...")
    start_time = time.time()
    try:
        results1 = service.generate_fast_recommendations(**test_case)
        time1 = time.time() - start_time
        print(f"완료: {time1:.3f}초, 결과: {len(results1)}개")
    except Exception as e:
        print(f"오류: {e}")
        time1 = 0
    
    # 두 번째 실행
    print("두 번째 실행 (캐시 있음)...")
    start_time = time.time()
    try:
        results2 = service.generate_fast_recommendations(**test_case)
        time2 = time.time() - start_time
        print(f"완료: {time2:.3f}초, 결과: {len(results2)}개")
    except Exception as e:
        print(f"오류: {e}")
        time2 = 0
    
    # 캐싱 효과
    if time1 > 0 and time2 > 0:
        if time2 < time1:
            improvement = ((time1 - time2) / time1) * 100
            print(f"캐싱 개선: {improvement:.1f}% 빨라짐")
            print(f"속도 향상: {time1/time2:.1f}배")
        else:
            print("캐싱 효과 없음")


if __name__ == "__main__":
    test_fast_vs_original()
    test_caching_effect()
