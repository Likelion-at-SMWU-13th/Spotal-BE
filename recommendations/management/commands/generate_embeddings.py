# 기존 Place 데이터에 대한 임베딩 생성 명령어

from django.core.management.base import BaseCommand
from django.db import transaction
from recommendations.models import Place, PlaceEmbedding
from recommendations.services.embedding_service import upsert_place_embedding


class Command(BaseCommand):
    help = '기존 Place 데이터에 대한 임베딩을 생성합니다'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='한 번에 처리할 배치 크기 (기본값: 10)'
        )
        parser.add_argument(
            '--force-update',
            action='store_true',
            help='기존 임베딩이 있어도 강제로 업데이트'
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        force_update = options['force_update']
        
        # 임베딩이 없는 Place들 찾기
        if force_update:
            places = Place.objects.all()
            self.stdout.write(f"전체 {places.count()}개 장소의 임베딩을 생성합니다...")
        else:
            places_without_embedding = Place.objects.filter(embedding__isnull=True)
            places = places_without_embedding
            self.stdout.write(f"임베딩이 없는 {places.count()}개 장소의 임베딩을 생성합니다...")
        
        if places.count() == 0:
            self.stdout.write(self.style.SUCCESS("처리할 장소가 없습니다."))
            return
        
        # 배치 처리
        total_processed = 0
        total_errors = 0
        
        for i in range(0, places.count(), batch_size):
            batch = places[i:i + batch_size]
            
            with transaction.atomic():
                for place in batch:
                    try:
                        if force_update or not hasattr(place, 'embedding'):
                            upsert_place_embedding(place)
                            total_processed += 1
                            self.stdout.write(f"✓ {place.name} 임베딩 생성 완료")
                        else:
                            self.stdout.write(f"- {place.name} 임베딩 이미 존재 (건너뜀)")
                            
                    except Exception as e:
                        total_errors += 1
                        self.stdout.write(
                            self.style.ERROR(f"✗ {place.name} 임베딩 생성 실패: {str(e)}")
                        )
            
            # 진행 상황 출력
            progress = min(i + batch_size, places.count())
            self.stdout.write(f"진행률: {progress}/{places.count()} ({progress/places.count()*100:.1f}%)")
        
        # 결과 요약
        self.stdout.write(
            self.style.SUCCESS(
                f"\n임베딩 생성 완료!\n"
                f"성공: {total_processed}개\n"
                f"실패: {total_errors}개"
            )
        )
