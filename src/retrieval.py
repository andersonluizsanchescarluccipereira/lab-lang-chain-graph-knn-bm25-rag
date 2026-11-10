"""Repositório OpenSearch: BM25, ANN k-NN e fusão por ranking."""
from typing import Any
from opensearchpy import OpenSearch
from . import config
from .catalog import opensearch_filters


def open_client() -> OpenSearch:
    return OpenSearch(
        hosts=[config.OPENSEARCH_URL],
        http_compress=True,
        timeout=60,
        max_retries=2,
        retry_on_timeout=True,
    )


def clean_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'id': hit['_id'], 'score': hit['_score'], **hit['_source']}
            for hit in response['hits']['hits']]


class VehicleSearch:
    def __init__(self, client: OpenSearch, embeddings: Any):
        self.client = client
        self.embeddings = embeddings

    def bm25(self, question: str, filters: dict[str, Any], size: int = 8):
        query = {
            'bool': {
                'must': [{'multi_match': {
                    'query': question,
                    'fields': ['marca^3', 'modelo^4', 'versao^2', 'texto_busca^2'],
                }}],
                'filter': opensearch_filters(filters),
            }
        }
        return clean_hits(self.client.search(
            index=config.OPENSEARCH_INDEX,
            body={'size': size, 'query': query, '_source': {'excludes': ['vetor']}},
        ))

    def vector(self, question: str, filters: dict[str, Any], size: int = 8):
        vector = self.embeddings.embed_query(question)
        knn: dict[str, Any] = {'vector': vector, 'k': size}
        clauses = opensearch_filters(filters)
        if clauses:
            knn['filter'] = {'bool': {'filter': clauses}}
        return clean_hits(self.client.search(
            index=config.OPENSEARCH_INDEX,
            body={'size': size,
                  'query': {'knn': {'vetor': knn}},
                  '_source': {'excludes': ['vetor']}},
        ))

