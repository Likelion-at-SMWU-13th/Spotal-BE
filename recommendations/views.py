from django.shortcuts import render
from rest_framework import status, generics, permissions
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.conf import settings
from .models import Place, SavedPlace, AISummary
from .serializers import *
from rest_framework.views import APIView
from search.service.summary_card import generate_summary_card, generate_emotion_tags
from search.service.address import translate_to_korean
from .services.google_service import get_similar_places, get_place_details, get_photo_url
from .services.utils import extract_neighborhood
from .services.emotion_service import expand_emotions_with_gpt   
from .services.rag_service import (
    RAGRecommendationService, 
    get_rag_recommendations,
    get_similar_places,
    get_personalized_recommendations
)
from .services.enhanced_recommendation_service import EnhancedRecommendationService
from .services.fast_hybrid_service import FastHybridService
from .services.ultra_fast_service import UltraFastService
from .services.smart_hybrid_service import SmartHybridService


# Create your views here.

# 추천가게 생성 
class RecommendationView(APIView):
    """추천 가게 생성 & 응답 API"""
    permission_classes = [AllowAny]

    def post(self, request):
        name = request.data.get("name")
        address = request.data.get("address")
        emotion_tags = request.data.get("emotion_tags", [])
        user_id = request.data.get("user_id", None)  # user_id 필드 optional

        # --- 필수 입력값 체크 ---
        if not name or not address or not emotion_tags:
            return Response(
                {"error": "name, address, emotion_tags는 필수 입력값입니다."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # --- 업태 구분 (카테고리) ---
        category_str = request.data.get("category", "")
        if "cafe" in category_str.lower():
            allowed_types = ["cafe"]
        else:
            allowed_types = ["restaurant", "food"]

        try:
            # 스마트 하이브리드 서비스 사용 (상황에 따라 RAG vs 구글맵 선택)
            smart_service = SmartHybridService()
            
            # 스마트 추천 생성
            recommendations = smart_service.generate_smart_recommendations(
                name=name,
                address=address,
                emotion_tags=emotion_tags,
                user_id=user_id,
                category=category_str,
                top_k=8
            )
            
            # 기존 API 형식으로 변환
            response_data = []
            for rec in recommendations:
                # Place 객체 가져오기
                try:
                    place = Place.objects.get(shop_id=rec['shop_id'])
                    response_data.append(PlaceSerializer(place).data)
                except Place.DoesNotExist:
                    # Place 객체가 없으면 새로 생성
                    location_obj = None
                    if rec.get('location'):
                        location_obj, _ = Location.objects.get_or_create(name=rec['location'])
                    
                    place = Place.objects.create(
                        name=rec['name'],
                        address=rec['address'],
                        photo_reference=rec.get('google_place_id', ''),
                        location=location_obj,
                        google_place_id=rec.get('google_place_id'),
                        google_rating=rec.get('google_rating', 0.0),
                        place_types=rec.get('place_types', []),
                        status=rec.get('status', 'operating')
                    )
                    
                    # 감정태그 설정
                    emotion_objs = []
                    for emotion_name in rec.get('emotions', []):
                        obj, _ = Emotion.objects.get_or_create(name=emotion_name)
                        emotion_objs.append(obj)
                    place.emotions.set(emotion_objs)
                    
                    # AI 요약 생성
                    AISummary.objects.create(shop=place, summary=rec.get('summary', ''))
                    
                    response_data.append(PlaceSerializer(place).data)
            
            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {"error": f"추천 생성 중 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )





class PlaceDetailView(generics.RetrieveAPIView):
    queryset = Place.objects.all()
    serializer_class = PlaceSerializer
    lookup_field = "shop_id"
    permission_classes = [permissions.AllowAny]



class SavedPlaceCreateView(generics.CreateAPIView):
    queryset = SavedPlace.objects.all()
    serializer_class = SavedPlaceCreateSerializer
    permission_classes = [permissions.AllowAny]

    # summary_snapshot에 최신 ai요약 저장해 놓기 
    def perform_create(self, serializer):
        saved_place = serializer.save()

        # 추천 로직에 따라 최신요약 저장 로직 분기
        last_summary = None
        if saved_place.rec == 1:
            last_summary = saved_place.shop.ai_summary.order_by("-created_date").first()
        elif saved_place.rec == 2:
            last_summary = saved_place.shop.infer_ai_summary.order_by("-created_date").first()

        if last_summary:
            saved_place.summary_snapshot = last_summary.summary
            saved_place.save()



class SavedPlaceListView(generics.ListAPIView):
    serializer_class = SavedPlaceSerializer
    permission_classes = [permissions.AllowAny]

    # user별 필터링해서 목록 보여줌. 
    def get_queryset(self):
        user_id = self.request.query_params.get("user")  # 쿼리 파라미터로 받기
        if user_id:
            return SavedPlace.objects.filter(user_id=user_id).order_by("-created_date")
        return SavedPlace.objects.all().order_by("-created_date")
        


class SavedPlaceDeleteView(generics.DestroyAPIView):
    serializer_class = SavedPlaceCreateSerializer
    permission_classes = [permissions.AllowAny]
    lookup_field = "saved_id"

    def get_queryset(self):
        return SavedPlace.objects.all()
    
    # 삭제되었다고 응답 띄우기
    def delete(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        data = serializer.data  
        self.perform_destroy(instance)
        return Response(
            {"message": "저장한 장소가 삭제되었습니다.", "deleted_place": data},
            status=status.HTTP_200_OK
        )



# AISummary만 따로 조회
class AISummaryDetailView(generics.RetrieveAPIView):
    serializer_class = AISummarySerializer
    permission_classes = [permissions.AllowAny]

    def get_object(self):
        shop_id = self.kwargs.get("shop_id")
        return AISummary.objects.get(shop__shop_id=shop_id) # 이건 추천1의 ai summary만 가져오는 코드임 
    

# AISummary만 따로 생성 (요약 재생성 요청 시 필요)
class AISummaryCreateUpdateView(generics.CreateAPIView):

    serializer_class = AISummarySerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, shop_id):
        try:
            place = Place.objects.get(shop_id=shop_id)
        except Place.DoesNotExist:
            return Response({"error": "해당 가게가 존재하지 않습니다."}, status=404)

        # GPT 요약 생성
        from .services import generate_gpt_emotion_based_recommendations
        summary_text = generate_gpt_emotion_based_recommendations(place)

        # 기존 요약 있으면 업데이트, 없으면 새로 생성
        aisummary, created = AISummary.objects.update_or_create(
            shop=place,
            defaults={"summary": summary_text}
        )

        return Response(
            {
                "message": "AI 요약 생성 완료" if created else "AI 요약 갱신 완료",
                "data": AISummarySerializer(aisummary).data
            },
            status=201 if created else 200
        )



class RAGSearchView(APIView):
    """RAG 기반 자연어 검색 API"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        query = request.data.get("query", "")
        user_id = request.data.get("user_id", None)
        emotion_filters = request.data.get("emotion_filters", [])
        location_filters = request.data.get("location_filters", [])
        top_k = request.data.get("top_k", 10)
        
        if not query:
            return Response(
                {"error": "검색 쿼리는 필수입니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user = None
            if user_id:
                from django.contrib.auth import get_user_model
                User = get_user_model()
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass
            
            places = get_rag_recommendations(
                query=query,
                user=user,
                emotion_filters=emotion_filters,
                location_filters=location_filters,
                top_k=top_k
            )
            
            serializer = PlaceSerializer(places, many=True)
            return Response({
                "query": query,
                "results": serializer.data,
                "count": len(places)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"검색 중 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SimilarPlacesView(APIView):
    """특정 장소와 유사한 장소 추천 API"""
    permission_classes = [AllowAny]
    
    def get(self, request, shop_id):
        user_id = request.query_params.get("user_id", None)
        top_k = int(request.query_params.get("top_k", 5))
        
        try:
            place = Place.objects.get(shop_id=shop_id)
            
            user = None
            if user_id:
                from django.contrib.auth import get_user_model
                User = get_user_model()
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass
            
            similar_places = get_similar_places(place, user, top_k)
            serializer = PlaceSerializer(similar_places, many=True)
            
            return Response({
                "place": PlaceSerializer(place).data,
                "similar_places": serializer.data,
                "count": len(similar_places)
            }, status=status.HTTP_200_OK)
            
        except Place.DoesNotExist:
            return Response(
                {"error": "해당 장소를 찾을 수 없습니다."},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"유사 장소 검색 중 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PersonalizedFeedView(APIView):
    """개인화된 추천 피드 API"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        user_id = request.query_params.get("user_id")
        top_k = int(request.query_params.get("top_k", 20))
        
        if not user_id:
            return Response(
                {"error": "user_id는 필수입니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            user = User.objects.get(id=user_id)
            
            recommendations = get_personalized_recommendations(user, top_k)
            serializer = PlaceSerializer(recommendations, many=True)
            
            return Response({
                "user_id": user_id,
                "recommendations": serializer.data,
                "count": len(recommendations)
            }, status=status.HTTP_200_OK)
            
        except User.DoesNotExist:
            return Response(
                {"error": "사용자를 찾을 수 없습니다."},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"개인화 추천 중 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EmotionBasedRAGView(APIView):
    """감정 기반 RAG 추천 API"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        emotions = request.data.get("emotions", [])
        location = request.data.get("location", None)
        user_id = request.data.get("user_id", None)
        top_k = request.data.get("top_k", 10)
        
        if not emotions:
            return Response(
                {"error": "감정 태그는 필수입니다."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user = None
            if user_id:
                from django.contrib.auth import get_user_model
                User = get_user_model()
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass
            
            service = RAGRecommendationService()
            results = service.get_emotion_based_recommendations(
                emotions=emotions,
                user=user,
                location=location,
                top_k=top_k
            )
            
            places = [place for place, _ in results]
            serializer = PlaceSerializer(places, many=True)
            
            return Response({
                "emotions": emotions,
                "location": location,
                "recommendations": serializer.data,
                "count": len(places)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"감정 기반 추천 중 오류 발생: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )