#!/usr/bin/env python
"""
RAG 성능 벤치마크 스크립트
사용법: python benchmark_rag_performance.py
"""

import os
import sys
import django
import time

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spotal.settings')
django.setup()

from recommendations.services.performance_monitor import RecommendationBenchmark
from recommendations.services.optimized_rag_service import OptimizedEnhancedService
from recommendations.services.enhanced_recommendation_service import EnhancedRecommendationService


def run_performance_comparison():
    """성능 비교 실행"""
    print("=== RAG 성능 벤치마크 시작 ===\n")
    
    benchmark = RecommendationBenchmark()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "스타벅스",
            "address": "용산구 이태원동",
            "emotion_tags": ["편안함", "조용함"],
            "user_id": 1
        },
        {
            "name": "맛있는 파스타집",
            "address": "용산구 한남동",
            "emotion_tags": ["로맨틱", "아늑함"],
            "user_id": 1
        },
        {
            "name": "조용한 카페",
            "address": "용산구 한강로",
            "emotion_tags": ["조용함", "집중"],
            "user_id": 1
        },
        {
            "name": "데이트 레스토랑",
            "address": "용산구 이촌동",
            "emotion_tags": ["로맨틱", "아름다움"],
            "user_id": 1
        }
    ]
    
    print(f"총 {len(test_cases)}개 테스트 케이스 실행 중...\n")
    
    # 성능 비교 실행
    results = benchmark.run_comparison(test_cases)
    
    # 결과 출력
    print("\n" + "="*50)
    print("성능 비교 결과")
    print("="*50)
    
    summary = results['summary']
    print(f"평균 기존 시스템 시간: {summary.get('avg_old_time', 0):.3f}초")
    print(f"평균 RAG 시스템 시간: {summary.get('avg_rag_time', 0):.3f}초")
    print(f"평균 개선율: {summary.get('avg_improvement', 0):.1f}%")
    print(f"총 테스트 수: {summary.get('total_tests', 0)}개")
    
    # 개별 결과
    print("\n개별 테스트 결과:")
    for i, (old, rag, comp) in enumerate(zip(
        results['old_system'], 
        results['rag_system'], 
        results['comparisons']
    )):
        print(f"테스트 {i+1}:")
        print(f"  기존: {old.get('execution_time', 0):.3f}s")
        print(f"  RAG: {rag.get('execution_time', 0):.3f}s")
        if comp:
            print(f"  개선: {comp.get('time_improvement_percent', 0):.1f}%")
        print()
    
    return results


def test_optimized_vs_original():
    """최적화된 버전 vs 원본 비교"""
    print("\n=== 최적화된 RAG vs 원본 RAG 비교 ===\n")
    
    test_cases = [
        {
            "name": "스타벅스",
            "address": "용산구 이태원동",
            "emotion_tags": ["편안함", "조용함"],
            "user_id": 1,
            "category": "cafe",
            "top_k": 8
        }
    ]
    
    # 원본 서비스
    original_service = EnhancedRecommendationService()
    optimized_service = OptimizedEnhancedService()
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}: {test_case['name']}")
        
        # 원본 테스트
        print("  원본 RAG 서비스 테스트...")
        start_time = time.time()
        try:
            original_results = original_service.generate_enhanced_recommendations(**test_case)
            original_time = time.time() - start_time
            print(f"    완료: {original_time:.3f}초, 결과: {len(original_results)}개")
        except Exception as e:
            print(f"    오류: {e}")
            original_time = 0
            original_results = []
        
        # 최적화된 서비스 테스트
        print("  최적화된 RAG 서비스 테스트...")
        start_time = time.time()
        try:
            optimized_results = optimized_service.generate_enhanced_recommendations_optimized(**test_case)
            optimized_time = time.time() - start_time
            print(f"    완료: {optimized_time:.3f}초, 결과: {len(optimized_results)}개")
        except Exception as e:
            print(f"    오류: {e}")
            optimized_time = 0
            optimized_results = []
        
        # 비교
        if original_time > 0 and optimized_time > 0:
            improvement = ((original_time - optimized_time) / original_time) * 100
            print(f"  개선율: {improvement:.1f}%")
            
            results.append({
                'original_time': original_time,
                'optimized_time': optimized_time,
                'improvement': improvement
            })
        print()
    
    # 전체 결과
    if results:
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        print(f"평균 개선율: {avg_improvement:.1f}%")
    
    return results


def test_caching_effectiveness():
    """캐싱 효과 테스트"""
    print("\n=== 캐싱 효과 테스트 ===\n")
    
    service = OptimizedEnhancedService()
    test_case = {
        "name": "스타벅스",
        "address": "용산구 이태원동",
        "emotion_tags": ["편안함", "조용함"],
        "user_id": 1,
        "category": "cafe",
        "top_k": 8
    }
    
    # 첫 번째 실행 (캐시 없음)
    print("첫 번째 실행 (캐시 없음)...")
    start_time = time.time()
    try:
        results1 = service.generate_enhanced_recommendations_optimized(**test_case)
        time1 = time.time() - start_time
        print(f"완료: {time1:.3f}초, 결과: {len(results1)}개")
    except Exception as e:
        print(f"오류: {e}")
        time1 = 0
    
    # 두 번째 실행 (캐시 있음)
    print("두 번째 실행 (캐시 있음)...")
    start_time = time.time()
    try:
        results2 = service.generate_enhanced_recommendations_optimized(**test_case)
        time2 = time.time() - start_time
        print(f"완료: {time2:.3f}초, 결과: {len(results2)}개")
    except Exception as e:
        print(f"오류: {e}")
        time2 = 0
    
    # 캐싱 효과
    if time1 > 0 and time2 > 0:
        cache_improvement = ((time1 - time2) / time1) * 100
        print(f"캐싱 개선율: {cache_improvement:.1f}%")
        print(f"속도 향상: {time1/time2:.1f}배")
    
    return {
        'first_run': time1,
        'second_run': time2,
        'cache_improvement': cache_improvement if time1 > 0 and time2 > 0 else 0
    }


def main():
    """메인 실행 함수"""
    print("RAG 성능 벤치마크 시작...\n")
    
    try:
        # 1. 기본 성능 비교
        comparison_results = run_performance_comparison()
        
        # 2. 최적화된 버전 비교
        optimization_results = test_optimized_vs_original()
        
        # 3. 캐싱 효과 테스트
        cache_results = test_caching_effectiveness()
        
        print("\n" + "="*50)
        print("전체 벤치마크 완료")
        print("="*50)
        
        # 최종 요약
        if comparison_results.get('summary'):
            summary = comparison_results['summary']
            print(f"RAG vs 기존 시스템: {summary.get('avg_improvement', 0):.1f}% 개선")
        
        if optimization_results:
            avg_opt_improvement = sum(r['improvement'] for r in optimization_results) / len(optimization_results)
            print(f"최적화 효과: {avg_opt_improvement:.1f}% 개선")
        
        if cache_results.get('cache_improvement', 0) > 0:
            print(f"캐싱 효과: {cache_results['cache_improvement']:.1f}% 개선")
        
    except Exception as e:
        print(f"벤치마크 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
