from django.urls import path
from . import views

app_name = 'recommendations'

urlpatterns = [
    # 추천 가게 (POST: 추천 생성 & 응답)
    path("", views.RecommendationView.as_view(), name="recommendation"),
    path("recommend-stores/", views.RecommendationView.as_view(), name="recommend-stores"),

    # 가게 상세 조회
    path("<int:shop_id>/", views.PlaceDetailView.as_view(), name="place-detail"),

    # 장소 보관
    path("saved/create/", views.SavedPlaceCreateView.as_view(), name="savedplace-create"), # 장소 보관하기
    path("saved/", views.SavedPlaceListView.as_view(), name="savedplace-list"), # 보관한 장소 목록 조회
    path("saved/<int:saved_id>/delete/", views.SavedPlaceDeleteView.as_view(), name="savedplace-delete"), # 보관한 장소 삭제

    # AISummary 
    path("<int:shop_id>/summary/", views.AISummaryDetailView.as_view(), name="aisummary-detail"), # ai 요약만 따로 확인 
    path("<int:shop_id>/summary/create/", views.AISummaryCreateUpdateView.as_view(), name="aisummary-create"), # ai 요약만 새로 생성
    
    # RAG 기반 추천 API
    path("rag/search/", views.RAGSearchView.as_view(), name="rag-search"), # 자연어 검색
    path("rag/similar/<int:shop_id>/", views.SimilarPlacesView.as_view(), name="similar-places"), # 유사 장소 추천
    path("rag/personalized/", views.PersonalizedFeedView.as_view(), name="personalized-feed"), # 개인화 추천
    path("rag/emotion-based/", views.EmotionBasedRAGView.as_view(), name="emotion-based-rag"), # 감정 기반 추천
]
