"""
DB 상태 확인 스크립트
"""

import os
import sys
import django

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spotal.settings')
django.setup()

from recommendations.models import Place, PlaceEmbedding, SavedPlace, AISummary
from community.models import Emotion, Location
from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


def check_db_status():
    """DB 상태 확인"""
    print("=== Spotal-BE DB 상태 확인 ===\n")
    
    # 1. 기본 통계
    print("기본 통계:")
    print(f"  전체 장소 수: {Place.objects.count()}")
    print(f"  임베딩이 있는 장소 수: {PlaceEmbedding.objects.count()}")
    print(f"  AI 요약이 있는 장소 수: {AISummary.objects.count()}")
    print(f"  저장된 장소 수: {SavedPlace.objects.count()}")
    print(f"  감정 태그 수: {Emotion.objects.count()}")
    print(f"  위치 수: {Location.objects.count()}")
    print(f"  사용자 수: {User.objects.count()}")
    
    # 2. 최근 생성된 장소들
    print("\n최근 생성된 장소들:")
    recent_places = Place.objects.order_by('-created_date')[:5]
    for place in recent_places:
        print(f"  - {place.name} ({place.address})")
        print(f"    생성일: {place.created_date}")
        print(f"    임베딩: {'있음' if hasattr(place, 'embedding') else '없음'}")
        print(f"    AI 요약: {'있음' if place.ai_summary.exists() else '없음'}")
        print()
    
    # 3. 사용자별 저장 장소
    print("사용자별 저장 장소:")
    users_with_saves = User.objects.filter(saved_places__isnull=False).distinct()
    for user in users_with_saves:
        save_count = SavedPlace.objects.filter(user=user).count()
        print(f"  - {user.username or user.email or user.id}: {save_count}개 저장")
    
    # 4. 감정 태그 분포
    print("\n감정 태그 분포:")
    emotion_counts = {}
    for place in Place.objects.all():
        for emotion in place.emotions.all():
            emotion_counts[emotion.name] = emotion_counts.get(emotion.name, 0) + 1
    
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    for emotion, count in sorted_emotions[:10]:
        print(f"  - {emotion}: {count}개")
    
    # 5. RAG 사용 가능 여부 판단
    print("\nRAG 사용 가능 여부:")
    
    # 사용자 저장 장소 체크
    users_with_3_saves = User.objects.annotate(
        save_count=models.Count('saved_places')
    ).filter(save_count__gte=3).count()
    print(f"  3개 이상 저장한 사용자: {users_with_3_saves}명")
    
    # 유사 장소 체크
    similar_places = Place.objects.filter(
        models.Q(name__icontains="스타벅스") | 
        models.Q(name__icontains="카페") |
        models.Q(address__icontains="용산")
    ).count()
    print(f"  유사 장소 수: {similar_places}개")
    
    # 임베딩 체크
    embedding_count = PlaceEmbedding.objects.count()
    print(f"  임베딩 데이터: {embedding_count}개")
    
    # RAG 사용 권장 여부
    if users_with_3_saves >= 1 or similar_places >= 2 or embedding_count >= 10:
        print("  RAG 사용 권장!")
    else:
        print("  RAG 사용 조건 미충족 (구글맵 모드 권장)")
    
    print("\n=== DB 상태 확인 완료 ===")


if __name__ == "__main__":
    check_db_status()
