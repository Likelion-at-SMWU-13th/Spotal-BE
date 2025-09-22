# 빠른 하이브리드 추천 서비스 (RAG를 보조로만 사용)

from typing import List, Dict, Optional
from django.core.cache import cache
from django.db.models import Q, Count
import time

from recommendations.services.google_service import get_similar_places, get_place_details, get_photo_url
from recommendations.services.emotion_service import expand_emotions_with_gpt
from search.service.summary_card import generate_summary_card, generate_emotion_tags
from search.service.address import translate_to_korean
from recommendations.services.utils import extract_neighborhood
from community.models import Emotion, Location
from recommendations.models import Place, AISummary


class FastHybridService:
    """빠른 하이브리드 추천 서비스 - RAG를 보조로만 사용"""
    
    def __init__(self):
        self.cache_timeout = 600  # 10분 캐시
    
    def generate_fast_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int] = None,
        category: str = "",
        top_k: int = 8
    ) -> List[Dict]:
        """빠른 추천 생성 - 구글맵 우선, RAG는 보조"""
        
        # 캐시 확인
        cache_key = f"fast_rec:{hash(f'{name}{address}{emotion_tags}{user_id}')}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            # 1. 구글맵에서 빠르게 검색 (기존 방식)
            google_results = self._get_google_places_fast(
                name, address, emotion_tags, category, top_k
            )
            
            # 2. RAG는 기존 DB 데이터에서만 빠르게 보완 (선택적)
            if len(google_results) < top_k:
                rag_results = self._get_rag_supplement_fast(
                    name, address, emotion_tags, user_id, top_k - len(google_results)
                )
                google_results.extend(rag_results)
            
            # 3. 사용자 저장 장소 제외
            if user_id:
                from recommendations.models import SavedPlace
                saved_shop_ids = SavedPlace.objects.filter(
                    user_id=user_id, rec=1
                ).values_list("shop_id", flat=True)
                google_results = [
                    result for result in google_results 
                    if result.get('shop_id') not in saved_shop_ids
                ]
            
            # 4. 결과 캐시
            cache.set(cache_key, google_results, self.cache_timeout)
            
            execution_time = time.time() - start_time
            print(f"빠른 추천 생성: {execution_time:.3f}s, 결과: {len(google_results)}개")
            
            return google_results[:top_k]
            
        except Exception as e:
            print(f"빠른 추천 생성 오류: {e}")
            return []
    
    def _get_google_places_fast(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        category: str,
        top_k: int
    ) -> List[Dict]:
        """빠른 구글맵 검색"""
        
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
            for c in candidate_places:
                try:
                    place_id = c.get("place_id")
                    place_name = c.get("name")
                    
                    # 장소 상세 정보
                    details = get_place_details(place_id, place_name)
                    reviews = [r["text"] for r in details.get("reviews", [])]
                    uptaenms = details.get("types", [])
                    
                    # 한국어 정규화
                    name_ko = translate_to_korean(details.get("name")) if details.get("name") else None
                    address_ko = translate_to_korean(details.get("formatted_address")) if details.get("formatted_address") else None
                    
                    # 이미지 URL
                    photo_ref = ""
                    if details.get("photos"):
                        photo_ref = details["photos"][0].get("photo_reference", "")
                    image_url = get_photo_url(photo_ref) if photo_ref else ""
                    
                    # GPT 요약 생성
                    if reviews:
                        summary = generate_summary_card(details, reviews, uptaenms) or "요약 준비중입니다"
                    else:
                        neighborhood = extract_neighborhood(address_ko or c.get("address"))
                        summary = f"{place_name}은 {neighborhood}에 위치한 가게입니다"
                    
                    # 감정태그 생성
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
                            "photo_reference": photo_ref,
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
                        'source': 'google'
                    })
                    
                except Exception as e:
                    print(f"구글맵 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"구글맵 검색 오류: {e}")
            return []
    
    def _get_rag_supplement_fast(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int],
        top_k: int
    ) -> List[Dict]:
        """빠른 RAG 보완 검색 (기존 DB 데이터만)"""
        
        try:
            # 기존 DB에서만 빠르게 검색
            from recommendations.models import PlaceEmbedding
            
            # 감정 필터
            emotion_objs = self._get_cached_emotions(emotion_tags)
            emotion_names = [e.name for e in emotion_objs]
            
            # 간단한 키워드 검색 (임베딩 없이)
            places = Place.objects.filter(
                Q(name__icontains=name) | Q(address__icontains=address)
            ).filter(
                emotions__name__in=emotion_names
            ).distinct()[:top_k]
            
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
                        'source': 'rag_db'
                    })
                    
                except Exception as e:
                    print(f"RAG 보완 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"RAG 보완 검색 오류: {e}")
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
