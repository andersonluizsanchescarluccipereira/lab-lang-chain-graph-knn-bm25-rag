# Testes de RRF isolados de OpenSearch/Ollama, com mocks simples de import.
from src.fusion import rrf_merge


def test_documento_em_ambas_as_listas_sobe_no_ranking():
    bm25 = [{'id': 'a'}, {'id': 'b'}]
    vetor = [{'id': 'b'}, {'id': 'c'}]
    result = rrf_merge(bm25, vetor, size=3)
    assert result[0]['id'] == 'b'
    assert len(result) == 3


def test_lista_vazia():
    assert rrf_merge([], []) == []
