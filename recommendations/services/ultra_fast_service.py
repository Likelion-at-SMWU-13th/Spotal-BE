# 초고속 추천 서비스 - API 호출 최소화

from typing import List, Dict, Optional
from django.core.cache import cache
from django.db.models import Q
import time

from recommendations.services.google_service import get_similar_places, get_place_details, get_photo_url
from recommendations.services.emotion_service import expand_emotions_with_gpt
from search.service.summary_card import generate_summary_card, generate_emotion_tags
from search.service.address import translate_to_korean
from recommendations.services.utils import extract_neighborhood
from community.models import Emotion, Location
from recommendations.models import Place, AISummary


class UltraFastService:
    """초고속 추천 서비스 - API 호출 최소화"""
    
    def __init__(self):
        self.cache_timeout = 1800  # 30분 캐시
    
    def generate_ultra_fast_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int] = None,
        category: str = "",
        top_k: int = 8
    ) -> List[Dict]:
        """초고속 추천 생성 - API 호출 최소화"""
        
        # 캐시 확인
        cache_key = f"ultra_fast:{hash(f'{name}{address}{emotion_tags}{user_id}')}"
        cached_result = cache.get(cache_key)
        if cached_result:
            print("캐시에서 결과 반환")
            return cached_result
        
        start_time = time.time()
        
        try:
            # 1. 구글맵에서 기본 검색만 (상세 정보 조회 없이)
            basic_results = self._get_basic_google_places(
                name, address, emotion_tags, category, top_k
            )
            
            # 2. 상세 정보는 3개만 조회 (나머지는 기본 정보로)
            detailed_results = self._add_details_to_top_places(
                basic_results[:3], emotion_tags
            )
            
            # 3. 나머지는 기본 정보로 채우기
            remaining_results = self._add_basic_info_to_places(
                basic_results[3:], emotion_tags
            )
            
            # 4. 결과 합치기
            all_results = detailed_results + remaining_results
            
            # 5. 사용자 저장 장소 제외
            if user_id:
                from recommendations.models import SavedPlace
                saved_shop_ids = SavedPlace.objects.filter(
                    user_id=user_id, rec=1
                ).values_list("shop_id", flat=True)
                all_results = [
                    result for result in all_results 
                    if result.get('shop_id') not in saved_shop_ids
                ]
            
            # 6. 결과 캐시
            cache.set(cache_key, all_results, self.cache_timeout)
            
            execution_time = time.time() - start_time
            print(f"초고속 추천 생성: {execution_time:.3f}s, 결과: {len(all_results)}개")
            
            return all_results[:top_k]
            
        except Exception as e:
            print(f"초고속 추천 생성 오류: {e}")
            return []
    
    def _get_basic_google_places(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        category: str,
        top_k: int
    ) -> List[Dict]:
        """기본 구글맵 검색 (상세 정보 조회 없이)"""
        
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
                    # 기본 정보만으로 Place 객체 생성
                    neighborhood_name = extract_neighborhood(c.get("address", ""))
                    location_obj, _ = Location.objects.get_or_create(name=neighborhood_name)
                    
                    place, created = Place.objects.update_or_create(
                        google_place_id=c.get("place_id"),
                        defaults={
                            "name": c.get("name"),
                            "address": c.get("address"),
                            "photo_reference": c.get("photo_reference", ""),
                            "location": location_obj,
                            "google_rating": 0.0,  # 기본값
                            "place_types": [],
                            "status": "operating"
                        }
                    )
                    
                    # 기본 감정태그 설정 (GPT 호출 없이)
                    default_emotions = self._get_default_emotions_for_category(category)
                    emotion_objs = []
                    for emotion_name in default_emotions:
                        obj, _ = Emotion.objects.get_or_create(name=emotion_name)
                        emotion_objs.append(obj)
                    place.emotions.set(emotion_objs)
                    
                    # 기본 요약 생성 (GPT 호출 없이)
                    basic_summary = f"{place.name}은 {place.address}에 위치한 {category}입니다"
                    if created:
                        AISummary.objects.create(shop=place, summary=basic_summary)
                    
                    results.append({
                        'shop_id': place.shop_id,
                        'name': place.name,
                        'address': place.address,
                        'image_url': get_photo_url(place.photo_reference) if place.photo_reference else "",
                        'summary': basic_summary,
                        'emotions': default_emotions,
                        'location': place.location.name if place.location else None,
                        'google_place_id': place.google_place_id,
                        'google_rating': place.google_rating,
                        'place_types': place.place_types or [],
                        'status': place.status,
                        'source': 'google_basic'
                    })
                    
                except Exception as e:
                    print(f"기본 구글맵 결과 처리 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"기본 구글맵 검색 오류: {e}")
            return []
    
    def _add_details_to_top_places(
        self,
        places: List[Dict],
        emotion_tags: List[str]
    ) -> List[Dict]:
        """상위 3개 장소에만 상세 정보 추가"""
        
        detailed_results = []
        
        for place_data in places:
            try:
                place_id = place_data.get('google_place_id')
                if not place_id:
                    detailed_results.append(place_data)
                    continue
                
                # 상세 정보 조회
                details = get_place_details(place_id, place_data.get('name'))
                reviews = [r["text"] for r in details.get("reviews", [])]
                uptaenms = details.get("types", [])
                
                # 한국어 정규화
                name_ko = translate_to_korean(details.get("name")) if details.get("name") else place_data.get('name')
                address_ko = translate_to_korean(details.get("formatted_address")) if details.get("formatted_address") else place_data.get('address')
                
                # Place 객체 업데이트
                place = Place.objects.get(shop_id=place_data['shop_id'])
                place.name = name_ko
                place.address = address_ko
                place.google_rating = details.get("rating", 0.0)
                place.place_types = uptaenms
                place.save()
                
                # GPT 요약 생성 (1번만)
                if reviews:
                    summary = generate_summary_card(details, reviews, uptaenms) or place_data.get('summary', '')
                else:
                    summary = place_data.get('summary', '')
                
                # AI 요약 업데이트
                summary_obj, _ = AISummary.objects.get_or_create(shop=place)
                summary_obj.summary = summary
                summary_obj.save()
                
                # 감정태그 생성 (1번만)
                tags = generate_emotion_tags(details, reviews, uptaenms) or place_data.get('emotions', [])
                emotion_objs = []
                for tag_name in tags:
                    obj, _ = Emotion.objects.get_or_create(name=tag_name)
                    emotion_objs.append(obj)
                place.emotions.set(emotion_objs)
                
                # 결과 업데이트
                place_data.update({
                    'name': name_ko,
                    'address': address_ko,
                    'summary': summary,
                    'emotions': tags,
                    'google_rating': place.google_rating,
                    'place_types': place.place_types,
                    'source': 'google_detailed'
                })
                
                detailed_results.append(place_data)
                
            except Exception as e:
                print(f"상세 정보 추가 오류: {e}")
                detailed_results.append(place_data)
                continue
        
        return detailed_results
    
    def _add_basic_info_to_places(
        self,
        places: List[Dict],
        emotion_tags: List[str]
    ) -> List[Dict]:
        """나머지 장소들은 기본 정보만"""
        
        return places  # 이미 기본 정보가 있음
    
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
