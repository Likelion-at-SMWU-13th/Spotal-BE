# 최적화된 RAG 서비스

from typing import List, Dict, Optional, Tuple
from django.core.cache import cache
from django.db.models import Prefetch
import time
import logging

from recommendations.services.rag_service import RAGRecommendationService
from recommendations.models import Place, PlaceEmbedding, SavedPlace

logger = logging.getLogger(__name__)


class OptimizedRAGService(RAGRecommendationService):
    """최적화된 RAG 서비스"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(model_name)
        self.cache_timeout = 300  # 5분 캐시
    
    def hybrid_search_optimized(
        self, 
        query: str, 
        user: Optional[object] = None,
        emotion_filters: Optional[List[str]] = None,
        location_filters: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[Place, float]]:
        """최적화된 하이브리드 검색"""
        
        # 1. 캐시 확인
        cache_key = f"rag_search:{hash(query)}:{user.id if user else 'anon'}:{top_k}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("캐시에서 결과 반환")
            return cached_result
        
        start_time = time.time()
        
        # 2. 쿼리 벡터화 (캐시)
        query_vector = self._get_cached_embedding(query)
        
        # 3. 최적화된 DB 쿼리
        embeddings_query = self._get_optimized_embeddings_query(
            emotion_filters, location_filters
        )
        
        # 4. 배치 유사도 계산
        results = self._batch_similarity_calculation(
            query_vector, embeddings_query, top_k
        )
        
        # 5. 사용자 가중치 적용 (캐시)
        if user:
            results = self._apply_cached_user_weights(user, results)
        
        # 6. 결과 캐시
        cache.set(cache_key, results, self.cache_timeout)
        
        execution_time = time.time() - start_time
        logger.info(f"RAG 검색 완료: {execution_time:.3f}s, 결과: {len(results)}개")
        
        return results
    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """캐시된 임베딩 가져오기"""
        cache_key = f"embedding:{hash(text)}"
        cached_embedding = cache.get(cache_key)
        
        if cached_embedding:
            return cached_embedding
        
        # 임베딩 생성 및 캐시
        embedding = embed_text(text, self.model_name)
        cache.set(cache_key, embedding, self.cache_timeout * 2)  # 더 긴 캐시
        
        return embedding
    
    def _get_optimized_embeddings_query(self, emotion_filters, location_filters):
        """최적화된 임베딩 쿼리"""
        from django.db.models import Q
        
        query = PlaceEmbedding.objects.select_related(
            'place__location'
        ).prefetch_related(
            'place__emotions'
        )
        
        if emotion_filters:
            query = query.filter(place__emotions__name__in=emotion_filters).distinct()
        
        if location_filters:
            query = query.filter(place__location__name__in=location_filters).distinct()
        
        return query
    
    def _batch_similarity_calculation(self, query_vector, embeddings_query, top_k):
        """배치 유사도 계산"""
        import numpy as np
        
        # 모든 임베딩을 한 번에 가져오기
        embeddings = list(embeddings_query)
        
        if not embeddings:
            return []
        
        # 벡터를 numpy 배열로 변환
        query_vec = np.array(query_vector)
        place_vectors = np.array([emb.vector for emb in embeddings])
        
        # 배치 코사인 유사도 계산
        similarities = np.dot(place_vectors, query_vec) / (
            np.linalg.norm(place_vectors, axis=1) * np.linalg.norm(query_vec)
        )
        
        # 상위 결과 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 경우만
                results.append((embeddings[idx].place, float(similarities[idx])))
        
        return results
    
    def _apply_cached_user_weights(self, user, results):
        """캐시된 사용자 가중치 적용"""
        cache_key = f"user_weights:{user.id}"
        user_weights = cache.get(cache_key)
        
        if not user_weights:
            user_weights = self._calculate_user_weights(user)
            cache.set(cache_key, user_weights, self.cache_timeout)
        
        # 가중치 적용
        weighted_results = []
        for place, score in results:
            weighted_score = score
            
            # 감정 가중치
            place_emotions = set(place.emotions.values_list('name', flat=True))
            for emotion, weight in user_weights.get('emotions', []):
                if emotion in place_emotions:
                    weighted_score += weight
            
            # 위치 가중치
            if place.location and place.location.name in user_weights.get('locations', {}):
                weighted_score += user_weights['locations'][place.location.name]
            
            weighted_results.append((place, weighted_score))
        
        # 가중치 적용 후 재정렬
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        return weighted_results
    
    def _calculate_user_weights(self, user):
        """사용자 가중치 계산"""
        from collections import defaultdict
        
        # 저장된 장소들 분석
        saved_places = SavedPlace.objects.filter(user=user).select_related('shop')
        
        emotion_counts = defaultdict(int)
        location_counts = defaultdict(int)
        
        for saved in saved_places:
            for emotion in saved.shop.emotions.all():
                emotion_counts[emotion.name] += 1
            
            if saved.shop.location:
                location_counts[saved.shop.location.name] += 1
        
        # 정규화된 가중치
        max_emotion_count = max(emotion_counts.values()) if emotion_counts else 1
        max_location_count = max(location_counts.values()) if location_counts else 1
        
        emotions = [(emotion, count / max_emotion_count * 0.1) 
                   for emotion, count in emotion_counts.items()]
        locations = {loc: count / max_location_count * 0.05 
                    for loc, count in location_counts.items()}
        
        return {
            'emotions': emotions,
            'locations': locations
        }


# 성능 최적화된 Enhanced 서비스
class OptimizedEnhancedService:
    """최적화된 Enhanced 추천 서비스"""
    
    def __init__(self):
        self.rag_service = OptimizedRAGService()
        self.cache_timeout = 300
    
    def generate_enhanced_recommendations_optimized(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int] = None,
        category: str = "",
        top_k: int = 8
    ) -> List[Dict]:
        """최적화된 추천 생성"""
        
        # 캐시 키 생성
        cache_key = f"enhanced_rec:{hash(f'{name}{address}{emotion_tags}{user_id}{category}')}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("캐시에서 추천 결과 반환")
            return cached_result
        
        start_time = time.time()
        
        try:
            # RAG 검색 (최적화됨)
            user = None
            if user_id:
                from django.contrib.auth import get_user_model
                User = get_user_model()
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass
            
            # 감정 확장 (캐시)
            emotions = self._get_cached_expanded_emotions(emotion_tags)
            emotion_names = [e.name for e in emotions]
            
            # RAG 검색
            query_parts = [name]
            if emotion_names:
                query_parts.append(f"감정: {', '.join(emotion_names)}")
            query_parts.append(f"지역: {address}")
            rag_query = " ".join(query_parts)
            
            rag_results = self.rag_service.hybrid_search_optimized(
                query=rag_query,
                user=user,
                emotion_filters=emotion_names,
                top_k=top_k
            )
            
            # 결과 변환
            results = []
            for place, score in rag_results:
                try:
                    summary_obj = place.ai_summary.order_by("-created_date").first()
                    summary = summary_obj.summary if summary_obj else f"{place.name}은 {place.address}에 위치한 가게입니다"
                    
                    place_emotions = list(place.emotions.values_list('name', flat=True))
                    
                    results.append({
                        'place': {
                            'shop_id': place.shop_id,
                            'name': place.name,
                            'address': place.address,
                            'summary': summary,
                            'emotions': place_emotions,
                            'location': place.location.name if place.location else None,
                            'google_place_id': place.google_place_id,
                            'google_rating': place.google_rating,
                            'place_types': place.place_types or [],
                            'status': place.status
                        },
                        'is_rag_result': True,
                        'rag_score': score,
                        'source': 'rag'
                    })
                except Exception as e:
                    logger.error(f"결과 변환 오류: {e}")
                    continue
            
            # 결과 캐시
            cache.set(cache_key, results, self.cache_timeout)
            
            execution_time = time.time() - start_time
            logger.info(f"최적화된 추천 생성 완료: {execution_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"최적화된 추천 생성 오류: {e}")
            return []
    
    def _get_cached_expanded_emotions(self, emotion_tags):
        """캐시된 감정 확장"""
        cache_key = f"expanded_emotions:{hash(tuple(emotion_tags))}"
        cached_emotions = cache.get(cache_key)
        
        if cached_emotions:
            from community.models import Emotion
            return Emotion.objects.filter(name__in=cached_emotions)
        
        # 감정 확장 실행
        from recommendations.services.emotion_service import expand_emotions_with_gpt
        emotions = expand_emotions_with_gpt(emotion_tags)
        
        # 결과 캐시
        emotion_names = [e.name for e in emotions]
        cache.set(cache_key, emotion_names, self.cache_timeout)
        
        return emotions
