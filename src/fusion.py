"""Combina rank BM25 e rank k-NN sem misturar escalas dos scores."""
from typing import Any


def rrf_merge(bm25: list[dict[str, Any]], vector: list[dict[str, Any]],
              size: int = 5, constant: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (não soma scores BM25 e k-NN diretamente)."""
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for result_set in (bm25, vector):
        for rank, item in enumerate(result_set, start=1):
            doc_id = item['id']
            docs[doc_id] = item
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (constant + rank)
    return [{**docs[id_], 'rrf_score': score} for id_, score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:size]]
