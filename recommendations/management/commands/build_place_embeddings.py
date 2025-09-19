from django.core.management.base import BaseCommand

from recommendations.models import Place
from recommendations.services.embedding_service import upsert_place_embedding


class Command(BaseCommand):
    help = "Build or refresh embeddings for all Places"

    def add_arguments(self, parser):
        parser.add_argument("--model", default="text-embedding-3-small")
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        model_name = options["model"]
        limit = options["limit"]

        qs = Place.objects.all().order_by("shop_id")
        if limit and limit > 0:
            qs = qs[:limit]

        total = qs.count()
        self.stdout.write(self.style.NOTICE(f"Building embeddings for {total} places using {model_name}"))

        for idx, place in enumerate(qs, start=1):
            upsert_place_embedding(place, model_name=model_name)
            self.stdout.write(self.style.SUCCESS(f"[{idx}/{total}] {place.name}"))

        self.stdout.write(self.style.SUCCESS("Done."))


