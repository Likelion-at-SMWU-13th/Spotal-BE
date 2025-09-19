from typing import List, Optional

from django.db import transaction

from recommendations.models import Place, AISummary, PlaceEmbedding
from recommendations.services.gpt_client import client


def build_corpus_for_place(place: Place) -> str:
    parts: List[str] = []
    parts.append(f"이름: {place.name}")
    parts.append(f"주소: {place.address}")

    # 감정 태그
    try:
        emotion_names = list(place.emotions.values_list("name", flat=True))
        if emotion_names:
            parts.append("감정태그: " + ", ".join(emotion_names))
    except Exception:
        pass

    # 최신 요약
    try:
        summary_obj: Optional[AISummary] = place.ai_summary.order_by("-created_date").first()
        if summary_obj and summary_obj.summary:
            parts.append("요약: " + summary_obj.summary)
    except Exception:
        pass

    # 리뷰 일부(길이 제한)
    try:
        reviews = place.reviews or []
        if isinstance(reviews, list) and reviews:
            sample_reviews = reviews[:5]
            joined = " \n".join([str(r) for r in sample_reviews])
            parts.append("리뷰 샘플: " + joined)
    except Exception:
        pass

    return "\n".join(parts)


def embed_text(text: str, model: str = "text-embedding-3-small") -> List[float]:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding  # type: ignore[no-any-return]


@transaction.atomic
def upsert_place_embedding(place: Place, model_name: str = "text-embedding-3-small") -> PlaceEmbedding:
    corpus = build_corpus_for_place(place)
    vector = embed_text(corpus, model=model_name)

    embedding_obj, _ = PlaceEmbedding.objects.update_or_create(
        place=place,
        defaults={
            "vector": vector,
            "source_text": corpus,
            "model_name": model_name,
        },
    )
    return embedding_obj


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    # 안전 가드
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def search_places(query: str, top_k: int = 10, model_name: str = "text-embedding-3-small") -> List[Place]:
    query_vec = embed_text(query, model=model_name)
    items = list(PlaceEmbedding.objects.select_related("place").all())
    scored = []
    for item in items:
        sim = cosine_similarity(query_vec, item.vector)
        scored.append((sim, item.place))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


