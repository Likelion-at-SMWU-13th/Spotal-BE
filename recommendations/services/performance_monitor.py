# 성능 모니터링 및 비교 도구

import time
import logging
from typing import Dict, List, Any
from django.db import connection
from django.conf import settings

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """타이머 시작"""
        self.metrics[operation] = {
            'start_time': time.time(),
            'db_queries_before': len(connection.queries)
        }
    
    def end_timer(self, operation: str) -> Dict[str, Any]:
        """타이머 종료 및 메트릭 반환"""
        if operation not in self.metrics:
            return {}
        
        end_time = time.time()
        start_data = self.metrics[operation]
        
        metrics = {
            'operation': operation,
            'execution_time': end_time - start_data['start_time'],
            'db_queries': len(connection.queries) - start_data['db_queries_before'],
            'timestamp': end_time
        }
        
        # 로깅
        logger.info(f"{operation}: {metrics['execution_time']:.3f}s, DB queries: {metrics['db_queries']}")
        
        return metrics
    
    def compare_performance(self, old_metrics: Dict, new_metrics: Dict) -> Dict[str, Any]:
        """성능 비교"""
        if not old_metrics or not new_metrics:
            return {}
        
        time_improvement = ((old_metrics['execution_time'] - new_metrics['execution_time']) / old_metrics['execution_time']) * 100
        query_improvement = old_metrics['db_queries'] - new_metrics['db_queries']
        
        return {
            'time_improvement_percent': time_improvement,
            'query_reduction': query_improvement,
            'old_time': old_metrics['execution_time'],
            'new_time': new_metrics['execution_time'],
            'old_queries': old_metrics['db_queries'],
            'new_queries': new_metrics['db_queries']
        }


class RecommendationBenchmark:
    """추천 시스템 벤치마크"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
    
    def benchmark_old_system(self, name: str, address: str, emotion_tags: List[str], user_id: int = None) -> Dict:
        """기존 시스템 벤치마크"""
        from recommendations.services.google_service import get_similar_places
        from recommendations.services.emotion_service import expand_emotions_with_gpt
        
        self.monitor.start_timer('old_system')
        
        try:
            # 기존 로직 시뮬레이션
            emotions = expand_emotions_with_gpt(emotion_tags)
            emotion_names = [e.name for e in emotions]
            
            candidate_places = get_similar_places(address, emotion_names)[:8]
            
            return self.monitor.end_timer('old_system')
        except Exception as e:
            logger.error(f"Old system benchmark error: {e}")
            return self.monitor.end_timer('old_system')
    
    def benchmark_rag_system(self, name: str, address: str, emotion_tags: List[str], user_id: int = None) -> Dict:
        """RAG 시스템 벤치마크"""
        from recommendations.services.enhanced_recommendation_service import EnhancedRecommendationService
        
        self.monitor.start_timer('rag_system')
        
        try:
            service = EnhancedRecommendationService()
            results = service.generate_enhanced_recommendations(
                name=name,
                address=address,
                emotion_tags=emotion_tags,
                user_id=user_id,
                top_k=8
            )
            
            return self.monitor.end_timer('rag_system')
        except Exception as e:
            logger.error(f"RAG system benchmark error: {e}")
            return self.monitor.end_timer('rag_system')
    
    def run_comparison(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """성능 비교 실행"""
        results = {
            'old_system': [],
            'rag_system': [],
            'comparisons': []
        }
        
        for i, test_case in enumerate(test_cases):
            print(f"테스트 케이스 {i+1}/{len(test_cases)}: {test_case['name']}")
            
            # 기존 시스템 테스트
            old_metrics = self.benchmark_old_system(**test_case)
            results['old_system'].append(old_metrics)
            
            # RAG 시스템 테스트
            rag_metrics = self.benchmark_rag_system(**test_case)
            results['rag_system'].append(rag_metrics)
            
            # 비교
            comparison = self.monitor.compare_performance(old_metrics, rag_metrics)
            results['comparisons'].append(comparison)
            
            print(f"  기존: {old_metrics.get('execution_time', 0):.3f}s")
            print(f"  RAG: {rag_metrics.get('execution_time', 0):.3f}s")
            if comparison:
                print(f"  개선: {comparison.get('time_improvement_percent', 0):.1f}%")
        
        # 전체 통계
        results['summary'] = self._calculate_summary(results)
        return results
    
    def _calculate_summary(self, results: Dict) -> Dict[str, Any]:
        """전체 통계 계산"""
        old_times = [r.get('execution_time', 0) for r in results['old_system'] if r]
        rag_times = [r.get('execution_time', 0) for r in results['rag_system'] if r]
        
        if not old_times or not rag_times:
            return {}
        
        return {
            'avg_old_time': sum(old_times) / len(old_times),
            'avg_rag_time': sum(rag_times) / len(rag_times),
            'avg_improvement': ((sum(old_times) - sum(rag_times)) / sum(old_times)) * 100,
            'total_tests': len(old_times)
        }


# 사용 예시
def run_performance_test():
    """성능 테스트 실행"""
    benchmark = RecommendationBenchmark()
    
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
        }
    ]
    
    results = benchmark.run_comparison(test_cases)
    
    print("\n=== 성능 비교 결과 ===")
    summary = results['summary']
    print(f"평균 기존 시스템 시간: {summary.get('avg_old_time', 0):.3f}s")
    print(f"평균 RAG 시스템 시간: {summary.get('avg_rag_time', 0):.3f}s")
    print(f"평균 개선율: {summary.get('avg_improvement', 0):.1f}%")
    
    return results
