# 스마트 하이브리드 서비스 - RAG + API 최적화의 균형

from typing import List, Dict, Optional
from django.core.cache import cache
from django.db.models import Q
import time
import hashlib

from recommendations.services.google_service import get_similar_places, get_place_details, get_photo_url
from recommendations.services.emotion_service import expand_emotions_with_gpt
from search.service.summary_card import generate_summary_card, generate_emotion_tags
from search.service.address import translate_to_korean
from recommendations.services.utils import extract_neighborhood
from community.models import Emotion, Location
from recommendations.models import Place, AISummary, PlaceEmbedding
from recommendations.services.rag_service import RAGRecommendationService


class SmartHybridService:
    """스마트 하이브리드 서비스 - RAG + API 최적화의 균형"""
    
    def __init__(self):
        self.cache_timeout = 1800  # 30분 캐시
        self.rag_service = RAGRecommendationService()
    
    def generate_smart_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int] = None,
        category: str = "",
        top_k: int = 8
    ) -> List[Dict]:
        """스마트 추천 생성 - 상황에 따라 RAG vs 구글맵 선택"""
        
        # 캐시 확인
        cache_key = f"smart_rec:{hash(f'{name}{address}{emotion_tags}{user_id}')}"
        cached_result = cache.get(cache_key)
        if cached_result:
            print("캐시에서 결과 반환")
            return cached_result
        
        start_time = time.time()
        
        try:
            # 1. 상황 판단: RAG 사용할지 구글맵 사용할지
            use_rag = self._should_use_rag(name, address, emotion_tags, user_id)
            
            if use_rag:
                print("RAG 모드 사용")
                results = self._get_rag_recommendations(
                    name, address, emotion_tags, user_id, top_k
                )
            else:
                print("구글맵 모드 사용")
                results = self._get_google_recommendations(
                    name, address, emotion_tags, category, top_k
                )
            
            # 2. 결과가 부족하면 보완
            if len(results) < top_k:
                supplement_results = self._get_supplement_recommendations(
                    name, address, emotion_tags, user_id, top_k - len(results)
                )
                results.extend(supplement_results)
            
            # 3. 사용자 저장 장소 제외
            if user_id:
                from recommendations.models import SavedPlace
                saved_shop_ids = SavedPlace.objects.filter(
                    user_id=user_id, rec=1
                ).values_list("shop_id", flat=True)
                results = [
                    result for result in results 
                    if result.get('shop_id') not in saved_shop_ids
                ]
            
            # 4. 결과 캐시
            cache.set(cache_key, results, self.cache_timeout)
            
            execution_time = time.time() - start_time
            print(f"스마트 추천 생성: {execution_time:.3f}s, 결과: {len(results)}개")
            
            return results[:top_k]
            
        except Exception as e:
            print(f"스마트 추천 생성 오류: {e}")
            return []
    
    def _should_use_rag(self, name: str, address: str, emotion_tags: List[str], user_id: Optional[int]) -> bool:
        """RAG 사용 여부 판단"""
        
        # 1. 사용자가 있고 저장된 장소가 많으면 RAG 사용
        if user_id:
            from recommendations.models import SavedPlace
            saved_count = SavedPlace.objects.filter(user_id=user_id).count()
            if saved_count >= 3:  # 3개 이상 저장했으면 개인화 가능
                return True
        
        # 2. 기존 DB에 유사한 장소가 있으면 RAG 사용
        similar_places = Place.objects.filter(
            Q(name__icontains=name) | Q(address__icontains=address)
        ).count()
        if similar_places >= 2:  # 2개 이상 유사 장소 있으면 RAG 사용
            return True
        
        # 3. 임베딩이 있는 장소가 충분하면 RAG 사용
        embedding_count = PlaceEmbedding.objects.count()
        if embedding_count >= 10:  # 10개 이상 임베딩 있으면 RAG 사용
            return True
        
        # 4. 그 외에는 구글맵 사용
        return False
    
    def _get_rag_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int],
        top_k: int
    ) -> List[Dict]:
        """RAG 기반 추천 (빠른 버전)"""
        
        try:
            # 사용자 정보
            user = None
            if user_id:
                from django.contrib.auth import get_user_model
                User = get_user_model()
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass
            
            # 감정 확장 (캐시)
            emotions = self._get_cached_emotions(emotion_tags)
            emotion_names = [e.name for e in emotions]
            
            # RAG 검색 (빠른 버전)
            query_parts = [name]
            if emotion_names:
                query_parts.append(f"감정: {', '.join(emotion_names)}")
            query_parts.append(f"지역: {address}")
            rag_query = " ".join(query_parts)
            
            # RAG 검색 (상위 5개만)
            rag_results = self.rag_service.hybrid_search(
                query=rag_query,
                user=user,
                emotion_filters=emotion_names,
                top_k=min(5, top_k)
            )
            
            # 결과 변환
            results = []
            for place, score in rag_results:
                try:
                    # 기존 요약 사용 (GPT 호출 없이)
                    summary_obj = place.ai_summary.order_by("-created_date").first()
                    if summary_obj:
                        summary = summary_obj.summary
                    else:
                        # 간단한 요약 생성 (GPT 호출 없이)
                        summary = f"{place.name}은 {place.address}에 위치한 가게입니다"
                    
                    place_emotions = list(place.emotions.values_list('name', flat=True))
                    image_url = get_photo_url(place.photo_reference) if place.photo_reference else ""
                    
                    results.append({
                        'shop_id': place.shop_id,
                        'name': place.name,
                        'address': place.address,
                        'image_url': image_url,
                        'summary': summary,
                        'emotions': place_emotions,
                        'location': place.location.name if place.location else None,
                        'google_place_id': place.google_place_id,
                        'google_rating': place.google_rating,
                        'place_types': place.place_types or [],
                        'status': place.status,
                        'source': 'rag',
                        'rag_score': score
                    })
                    
                except Exception as e:
                    print(f"RAG 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"RAG 추천 오류: {e}")
            return []
    
    def _get_google_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        category: str,
        top_k: int
    ) -> List[Dict]:
        """구글맵 기반 추천 (최적화된 버전)"""
        
        try:
            # 감정 확장 (캐시)
            emotions = self._get_cached_emotions(emotion_tags)
            emotion_names = [e.name for e in emotions]
            
            # 업태 구분
            if "cafe" in category.lower():
                allowed_types = ["cafe"]
            else:
                allowed_types = ["restaurant", "food"]
            
            # 구글맵 검색
            candidate_places = get_similar_places(
                address, emotion_names, allowed_types=allowed_types
            )[:top_k]
            
            results = []
            for i, c in enumerate(candidate_places):
                try:
                    place_id = c.get("place_id")
                    place_name = c.get("name")
                    
                    # 상위 3개만 상세 정보 조회
                    if i < 3:
                        details = get_place_details(place_id, place_name)
                        reviews = [r["text"] for r in details.get("reviews", [])]
                        uptaenms = details.get("types", [])
                        
                        # 한국어 정규화
                        name_ko = translate_to_korean(details.get("name")) if details.get("name") else None
                        address_ko = translate_to_korean(details.get("formatted_address")) if details.get("formatted_address") else None
                        
                        # GPT 요약 생성 (상위 3개만)
                        if reviews:
                            summary = generate_summary_card(details, reviews, uptaenms) or "요약 준비중입니다"
                        else:
                            neighborhood = extract_neighborhood(address_ko or c.get("address"))
                            summary = f"{place_name}은 {neighborhood}에 위치한 가게입니다"
                        
                        # 감정태그 생성 (상위 3개만)
                        tags = generate_emotion_tags(details, reviews, uptaenms) or []
                        
                        # Location 매핑
                        neighborhood_name = extract_neighborhood(address_ko or c.get("address"))
                        location_obj, _ = Location.objects.get_or_create(name=neighborhood_name)
                        
                        # Place 객체 생성/업데이트
                        place, created = Place.objects.update_or_create(
                            google_place_id=place_id,
                            defaults={
                                "name": name_ko or place_name,
                                "address": address_ko or c.get("address"),
                                "photo_reference": c.get("photo_reference", ""),
                                "location": location_obj,
                                "google_rating": details.get("rating", 0.0),
                                "reviews": reviews,
                                "place_types": uptaenms
                            }
                        )
                        
                        # 감정태그 설정
                        emotion_objs = []
                        for tag_name in tags:
                            obj, _ = Emotion.objects.get_or_create(name=tag_name)
                            emotion_objs.append(obj)
                        place.emotions.set(emotion_objs)
                        
                        # AI 요약 생성
                        if created:
                            AISummary.objects.create(shop=place, summary=summary)
                        
                        # 임베딩 생성 (RAG를 위해)
                        try:
                            from recommendations.services.embedding_service import upsert_place_embedding
                            upsert_place_embedding(place)
                        except Exception as e:
                            print(f"임베딩 생성 중 오류: {e}")
                        
                        image_url = get_photo_url(place.photo_reference) if place.photo_reference else ""
                        source = 'google_detailed'
                        
                    else:
                        # 나머지는 기본 정보만
                        neighborhood_name = extract_neighborhood(c.get("address", ""))
                        location_obj, _ = Location.objects.get_or_create(name=neighborhood_name)
                        
                        place, created = Place.objects.update_or_create(
                            google_place_id=place_id,
                            defaults={
                                "name": place_name,
                                "address": c.get("address"),
                                "photo_reference": c.get("photo_reference", ""),
                                "location": location_obj,
                                "google_rating": 0.0,
                                "place_types": [],
                                "status": "operating"
                            }
                        )
                        
                        # 기본 감정태그
                        default_emotions = self._get_default_emotions_for_category(category)
                        emotion_objs = []
                        for emotion_name in default_emotions:
                            obj, _ = Emotion.objects.get_or_create(name=emotion_name)
                            emotion_objs.append(obj)
                        place.emotions.set(emotion_objs)
                        
                        # 기본 요약
                        summary = f"{place.name}은 {place.address}에 위치한 {category}입니다"
                        if created:
                            AISummary.objects.create(shop=place, summary=summary)
                        
                        tags = default_emotions
                        image_url = get_photo_url(place.photo_reference) if place.photo_reference else ""
                        source = 'google_basic'
                    
                    results.append({
                        'shop_id': place.shop_id,
                        'name': place.name,
                        'address': place.address,
                        'image_url': image_url,
                        'summary': summary,
                        'emotions': tags,
                        'location': place.location.name if place.location else None,
                        'google_place_id': place.google_place_id,
                        'google_rating': place.google_rating,
                        'place_types': place.place_types or [],
                        'status': place.status,
                        'source': source
                    })
                    
                except Exception as e:
                    print(f"구글맵 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"구글맵 추천 오류: {e}")
            return []
    
    def _get_supplement_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int],
        needed_count: int
    ) -> List[Dict]:
        """보완 추천 (기존 DB에서 키워드 검색)"""
        
        try:
            # 기존 DB에서 키워드 검색
            places = Place.objects.filter(
                Q(name__icontains=name) | Q(address__icontains=address)
            )[:needed_count]
            
            results = []
            for place in places:
                try:
                    summary_obj = place.ai_summary.order_by("-created_date").first()
                    summary = summary_obj.summary if summary_obj else f"{place.name}은 {place.address}에 위치한 가게입니다"
                    
                    place_emotions = list(place.emotions.values_list('name', flat=True))
                    image_url = get_photo_url(place.photo_reference) if place.photo_reference else ""
                    
                    results.append({
                        'shop_id': place.shop_id,
                        'name': place.name,
                        'address': place.address,
                        'image_url': image_url,
                        'summary': summary,
                        'emotions': place_emotions,
                        'location': place.location.name if place.location else None,
                        'google_place_id': place.google_place_id,
                        'google_rating': place.google_rating,
                        'place_types': place.place_types or [],
                        'status': place.status,
                        'source': 'db_supplement'
                    })
                    
                except Exception as e:
                    print(f"보완 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"보완 추천 오류: {e}")
            return []
    
    def _get_cached_emotions(self, emotion_tags: List[str]):
        """캐시된 감정 확장"""
        cache_key = f"emotions:{hash(tuple(emotion_tags))}"
        cached_emotions = cache.get(cache_key)
        
        if cached_emotions:
            return Emotion.objects.filter(name__in=cached_emotions)
        
        # 감정 확장 실행
        emotions = expand_emotions_with_gpt(emotion_tags)
        
        # 결과 캐시
        emotion_names = [e.name for e in emotions]
        cache.set(cache_key, emotion_names, self.cache_timeout)
        
        return emotions
    
    def _get_default_emotions_for_category(self, category: str) -> List[str]:
        """카테고리별 기본 감정태그"""
        if "cafe" in category.lower():
            return ["편안함", "조용함"]
        else:
            return ["맛있음", "정겨움"]
