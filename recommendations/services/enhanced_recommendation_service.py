from typing import List, Dict, Optional, Tuple
from django.contrib.auth import get_user_model
from django.db.models import Q, F, Count

from recommendations.models import Place, PlaceEmbedding, SavedPlace, AISummary
from recommendations.services.rag_service import RAGRecommendationService
from recommendations.services.google_service import get_similar_places, get_place_details, get_photo_url
from recommendations.services.emotion_service import expand_emotions_with_gpt
from search.service.summary_card import generate_summary_card, generate_emotion_tags
from search.service.address import translate_to_korean, normalize_korean_address
from recommendations.services.utils import extract_neighborhood
from community.models import Emotion, Location

User = get_user_model()


class EnhancedRecommendationService:
    """기존 추천 시스템에 RAG를 통합한 개선된 서비스"""
    
    def __init__(self):
        self.rag_service = RAGRecommendationService()
    
    def generate_enhanced_recommendations(
        self,
        name: str,
        address: str,
        emotion_tags: List[str],
        user_id: Optional[int] = None,
        category: str = "",
        top_k: int = 8
    ) -> List[Dict]:
        """
        기존 API 인터페이스를 유지하면서 RAG를 적용한 개선된 추천 생성
        """
        
        # 1. GPT 기반 감정 확장
        emotions = expand_emotions_with_gpt(emotion_tags)
        emotion_names = [e.name for e in emotions]
        
        # 2. 사용자 정보 가져오기
        user = None
        if user_id:
            try:
                user = User.objects.get(id=user_id)
            except User.DoesNotExist:
                pass
        
        # 3. RAG 기반 검색 시도
        rag_results = self._try_rag_search(
            name, address, emotion_names, user, top_k
        )
        
        # 4. RAG 결과가 충분하지 않으면 구글맵으로 보완
        if len(rag_results) < top_k:
            google_results = self._get_google_places(
                address, emotion_names, category, top_k - len(rag_results)
            )
            rag_results.extend(google_results)
        
        # 5. 사용자 저장 장소 제외 필터링
        if user_id:
            saved_shop_ids = SavedPlace.objects.filter(
                user_id=user_id, rec=1
            ).values_list("shop_id", flat=True)
            rag_results = [
                result for result in rag_results 
                if result.get('place', {}).get('shop_id') not in saved_shop_ids
            ]
        
        # 6. 최종 결과 반환
        return rag_results[:top_k]
    
    def _try_rag_search(
        self,
        name: str,
        address: str,
        emotion_names: List[str],
        user: Optional[User],
        top_k: int
    ) -> List[Dict]:
        """RAG 기반 검색 시도"""
        
        try:
            # 자연어 쿼리 구성
            query_parts = [name]
            if emotion_names:
                query_parts.append(f"감정: {', '.join(emotion_names)}")
            query_parts.append(f"지역: {address}")
            rag_query = " ".join(query_parts)
            
            # RAG 검색 실행
            rag_results = self.rag_service.hybrid_search(
                query=rag_query,
                user=user,
                emotion_filters=emotion_names,
                top_k=top_k
            )
            
            # RAG 결과를 기존 API 형식으로 변환
            results = []
            for place, score in rag_results:
                try:
                    # 기존 요약 가져오기
                    summary_obj = place.ai_summary.order_by("-created_date").first()
                    summary = summary_obj.summary if summary_obj else f"{place.name}은 {place.address}에 위치한 가게입니다"
                    
                    # 감정태그 가져오기
                    place_emotions = list(place.emotions.values_list('name', flat=True))
                    
                    # 이미지 URL 생성
                    image_url = ""
                    if place.photo_reference:
                        image_url = get_photo_url(place.photo_reference)
                    
                    results.append({
                        'place': {
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
                            'status': place.status
                        },
                        'is_rag_result': True,
                        'rag_score': score,
                        'source': 'rag'
                    })
                    
                except Exception as e:
                    print(f"RAG 결과 처리 중 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"RAG 검색 중 오류: {e}")
            return []
    
    def _get_google_places(
        self,
        address: str,
        emotion_names: List[str],
        category: str,
        top_k: int
    ) -> List[Dict]:
        """구글맵 API를 통한 장소 검색"""
        
        try:
            # 업태 구분
            if "cafe" in category.lower():
                allowed_types = ["cafe"]
            else:
                allowed_types = ["restaurant", "food"]
            
            # 구글맵에서 장소 검색
            candidate_places = get_similar_places(
                address, emotion_names, allowed_types=allowed_types
            )[:top_k]
            
            results = []
            for c in candidate_places:
                try:
                    place_id = c.get("place_id")
                    place_name = c.get("name")
                    
                    # 장소 상세 정보 가져오기
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
                    
                    # 임베딩 생성 (RAG를 위해)
                    try:
                        from recommendations.services.embedding_service import upsert_place_embedding
                        upsert_place_embedding(place)
                    except Exception as e:
                        print(f"임베딩 생성 중 오류: {e}")
                    
                    results.append({
                        'place': {
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
                            'status': place.status
                        },
                        'is_rag_result': False,
                        'rag_score': 0.0,
                        'source': 'google'
                    })
                    
                except Exception as e:
                    print(f"구글맵 결과 처리 중 오류: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"구글맵 검색 중 오류: {e}")
            return []
    
    def get_hybrid_recommendations(
        self,
        query: str,
        user_id: Optional[int] = None,
        emotion_filters: Optional[List[str]] = None,
        location_filters: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """하이브리드 추천 (RAG + 구글맵)"""
        
        user = None
        if user_id:
            try:
                user = User.objects.get(id=user_id)
            except User.DoesNotExist:
                pass
        
        # RAG 검색
        rag_results = self.rag_service.hybrid_search(
            query=query,
            user=user,
            emotion_filters=emotion_filters,
            location_filters=location_filters,
            top_k=top_k
        )
        
        # 결과를 기존 API 형식으로 변환
        results = []
        for place, score in rag_results:
            try:
                summary_obj = place.ai_summary.order_by("-created_date").first()
                summary = summary_obj.summary if summary_obj else f"{place.name}은 {place.address}에 위치한 가게입니다"
                
                place_emotions = list(place.emotions.values_list('name', flat=True))
                image_url = get_photo_url(place.photo_reference) if place.photo_reference else ""
                
                results.append({
                    'place': {
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
                        'status': place.status
                    },
                    'is_rag_result': True,
                    'rag_score': score,
                    'source': 'rag'
                })
                
            except Exception as e:
                print(f"하이브리드 결과 처리 중 오류: {e}")
                continue
        
        return results
