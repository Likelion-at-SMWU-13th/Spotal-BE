# RAG 기반 추천 서비스
from typing import List, Dict, Optional, Tuple
from django.db import models
from django.contrib.auth import get_user_model
from django.db.models import Q, F, Count
import numpy as np
from collections import defaultdict

from recommendations.models import Place, PlaceEmbedding, SavedPlace, AISummary
from recommendations.services.embedding_service import (
    embed_text, cosine_similarity, build_corpus_for_place
)
from community.models import Emotion, Location

User = get_user_model()


class RAGRecommendationService:
    """RAG 기반 추천 서비스"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
    
    def build_user_context(self, user: User) -> Dict:
        """사용자 컨텍스트 정보 구축"""
        context = {
            'saved_places': [],
            'preferred_emotions': [],
            'location_preferences': [],
            'recent_activity': []
        }
        
        # 저장된 장소들
        saved_places = SavedPlace.objects.filter(user=user).select_related('shop')
        for saved in saved_places:
            context['saved_places'].append({
                'place': saved.shop,
                'emotions': list(saved.shop.emotions.values_list('name', flat=True)),
                'location': saved.shop.location.name if saved.shop.location else None,
                'saved_date': saved.created_date
            })
        
        # 선호 감정 분석
        emotion_counts = defaultdict(int)
        for saved in saved_places:
            for emotion in saved.shop.emotions.all():
                emotion_counts[emotion.name] += 1
        
        context['preferred_emotions'] = sorted(
            emotion_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # 위치 선호도
        location_counts = defaultdict(int)
        for saved in saved_places:
            if saved.shop.location:
                location_counts[saved.shop.location.name] += 1
        
        context['location_preferences'] = sorted(
            location_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return context
    
    def build_enhanced_query(self, query: str, user_context: Dict) -> str:
        """사용자 컨텍스트를 포함한 향상된 쿼리 구축"""
        enhanced_parts = [query]
        
        # 선호 감정 추가
        if user_context['preferred_emotions']:
            top_emotions = [emotion for emotion, _ in user_context['preferred_emotions'][:3]]
            enhanced_parts.append(f"감정: {', '.join(top_emotions)}")
        
        # 선호 위치 추가
        if user_context['location_preferences']:
            top_locations = [loc for loc, _ in user_context['location_preferences'][:2]]
            enhanced_parts.append(f"지역: {', '.join(top_locations)}")
        
        return " ".join(enhanced_parts)
    
    def hybrid_search(
        self, 
        query: str, 
        user: Optional[User] = None,
        emotion_filters: Optional[List[str]] = None,
        location_filters: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[Place, float]]:
        """하이브리드 검색 (키워드 + 벡터 + 필터링)"""
        
        # 1. 기본 벡터 검색
        query_vector = embed_text(query, self.model_name)
        
        # 2. 사용자 컨텍스트 적용
        if user:
            user_context = self.build_user_context(user)
            enhanced_query = self.build_enhanced_query(query, user_context)
            query_vector = embed_text(enhanced_query, self.model_name)
        
        # 3. 필터링 조건 적용
        embeddings_query = PlaceEmbedding.objects.select_related('place')
        
        if emotion_filters:
            embeddings_query = embeddings_query.filter(
                place__emotions__name__in=emotion_filters
            ).distinct()
        
        if location_filters:
            embeddings_query = embeddings_query.filter(
                place__location__name__in=location_filters
            ).distinct()
        
        # 4. 유사도 계산
        results = []
        for embedding in embeddings_query:
            similarity = cosine_similarity(query_vector, embedding.vector)
            
            # 5. 추가 가중치 적용
            weighted_score = similarity
            
            # 사용자 선호도 가중치
            if user:
                user_context = self.build_user_context(user)
                
                # 선호 감정 가중치
                place_emotions = set(embedding.place.emotions.values_list('name', flat=True))
                for emotion, count in user_context['preferred_emotions']:
                    if emotion in place_emotions:
                        weighted_score += 0.1 * (count / 10)  # 정규화된 가중치
                
                # 선호 위치 가중치
                if embedding.place.location:
                    for location, count in user_context['location_preferences']:
                        if embedding.place.location.name == location:
                            weighted_score += 0.05 * (count / 10)
            
            results.append((embedding.place, weighted_score))
        
        # 6. 정렬 및 상위 결과 반환
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_semantic_recommendations(
        self,
        place: Place,
        user: Optional[User] = None,
        top_k: int = 5
    ) -> List[Tuple[Place, float]]:
        """특정 장소와 유사한 장소 추천"""
        
        # 장소의 임베딩 가져오기
        try:
            place_embedding = PlaceEmbedding.objects.get(place=place)
            place_vector = place_embedding.vector
        except PlaceEmbedding.DoesNotExist:
            # 임베딩이 없으면 생성
            from recommendations.services.embedding_service import upsert_place_embedding
            place_embedding = upsert_place_embedding(place, self.model_name)
            place_vector = place_embedding.vector
        
        # 유사 장소 검색
        query = f"이름: {place.name} 주소: {place.address}"
        if user:
            user_context = self.build_user_context(user)
            query = self.build_enhanced_query(query, user_context)
        
        return self.hybrid_search(query, user, top_k=top_k)
    
    def get_emotion_based_recommendations(
        self,
        emotions: List[str],
        user: Optional[User] = None,
        location: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[Place, float]]:
        """감정 기반 추천"""
        
        query = f"감정: {', '.join(emotions)}"
        if location:
            query += f" 지역: {location}"
        
        return self.hybrid_search(
            query, 
            user, 
            emotion_filters=emotions,
            location_filters=[location] if location else None,
            top_k=top_k
        )
    
    def get_personalized_feed(
        self,
        user: User,
        top_k: int = 20
    ) -> List[Tuple[Place, float]]:
        """개인화된 피드 생성"""
        
        user_context = self.build_user_context(user)
        
        if not user_context['saved_places']:
            # 저장된 장소가 없으면 인기 장소 추천
            return self.get_trending_places(top_k)
        
        # 사용자 선호도 기반 추천
        recommendations = []
        
        # 1. 선호 감정 기반 추천
        if user_context['preferred_emotions']:
            top_emotions = [emotion for emotion, _ in user_context['preferred_emotions'][:3]]
            emotion_recs = self.get_emotion_based_recommendations(
                top_emotions, user, top_k=top_k//2
            )
            recommendations.extend(emotion_recs)
        
        # 2. 선호 위치 기반 추천
        if user_context['location_preferences']:
            top_locations = [loc for loc, _ in user_context['location_preferences'][:2]]
            location_recs = self.hybrid_search(
                f"지역: {', '.join(top_locations)}",
                user,
                location_filters=top_locations,
                top_k=top_k//2
            )
            recommendations.extend(location_recs)
        
        # 3. 중복 제거 및 정렬
        seen_places = set()
        unique_recommendations = []
        
        for place, score in sorted(recommendations, key=lambda x: x[1], reverse=True):
            if place.shop_id not in seen_places:
                seen_places.add(place.shop_id)
                unique_recommendations.append((place, score))
        
        return unique_recommendations[:top_k]
    
    def get_trending_places(self, top_k: int = 10) -> List[Tuple[Place, float]]:
        """인기 장소 추천 (저장 횟수 기반)"""
        
        trending = SavedPlace.objects.values('shop').annotate(
            save_count=Count('shop')
        ).order_by('-save_count')[:top_k]
        
        results = []
        for item in trending:
            try:
                place = Place.objects.get(shop_id=item['shop'])
                score = item['save_count'] / 10.0  # 정규화
                results.append((place, score))
            except Place.DoesNotExist:
                continue
        
        return results


# 편의 함수들
def get_rag_recommendations(
    query: str,
    user: Optional[User] = None,
    emotion_filters: Optional[List[str]] = None,
    location_filters: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Place]:
    """RAG 기반 추천 결과 반환"""
    service = RAGRecommendationService()
    results = service.hybrid_search(
        query, user, emotion_filters, location_filters, top_k
    )
    return [place for place, _ in results]


def get_similar_places(place: Place, user: Optional[User] = None, top_k: int = 5) -> List[Place]:
    """특정 장소와 유사한 장소들 반환"""
    service = RAGRecommendationService()
    results = service.get_semantic_recommendations(place, user, top_k)
    return [place for place, _ in results]


def get_personalized_recommendations(user: User, top_k: int = 20) -> List[Place]:
    """개인화된 추천 결과 반환"""
    service = RAGRecommendationService()
    results = service.get_personalized_feed(user, top_k)
    return [place for place, _ in results]
