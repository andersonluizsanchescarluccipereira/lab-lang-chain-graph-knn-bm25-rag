"""Cria o índice BM25 + vetor e importa catálogo fictício.

Uso: python index.py [--reset]
--reset apaga SOMENTE o índice configurado (lab_veiculos_rag por padrão).
"""
import argparse
import json
from opensearchpy import helpers
from langchain_ollama import OllamaEmbeddings
from src import config
from src.catalog import build_search_text
from src.retrieval import open_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true',
                        help='Excluir o índice deste laboratório antes de recriar')
    args = parser.parse_args()
    client = open_client()
    if not client.ping():
        raise SystemExit('OpenSearch indisponível. Execute podman compose up -d e verifique porta 9200.')
    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL,
                                  base_url=config.OLLAMA_BASE_URL)
    path = config.ROOT / 'data' / 'vehicles.json'
    vehicles = json.loads(path.read_text(encoding='utf-8'))
    texts = [build_search_text(v) for v in vehicles]
    print(f'Gerando {len(texts)} embeddings localmente via Ollama ({config.OLLAMA_EMBED_MODEL})...')
    vectors = embeddings.embed_documents(texts)
    dimension = len(vectors[0])
    print(f'Dimensões do embedding: {dimension}')

    if args.reset and client.indices.exists(index=config.OPENSEARCH_INDEX):
        client.indices.delete(index=config.OPENSEARCH_INDEX)
        print(f'Índice {config.OPENSEARCH_INDEX} excluído explicitamente (--reset).')

    if client.indices.exists(index=config.OPENSEARCH_INDEX):
        mapping = client.indices.get_mapping(index=config.OPENSEARCH_INDEX)
        current_dimension = mapping[config.OPENSEARCH_INDEX]['mappings']['properties']['vetor']['dimension']
        if int(current_dimension) != dimension:
            raise SystemExit(
                f'Índice existente usa dimensão {current_dimension}, mas o modelo gera {dimension}. '
                'Se deseja substituir SOMENTE este índice de laboratório, execute python index.py --reset'
            )
    else:
        client.indices.create(index=config.OPENSEARCH_INDEX, body={
            'settings': {'index': {'knn': True, 'number_of_shards': 1,
                                   'number_of_replicas': 0}},
            'mappings': {'properties': {
                'marca': {'type': 'text', 'fields': {'raw': {'type': 'keyword'}}},
                'modelo': {'type': 'text'},
                'versao': {'type': 'text'},
                'ano': {'type': 'integer'},
                'preco': {'type': 'integer'},
                'km': {'type': 'integer'},
                'carroceria': {'type': 'keyword'},
                'cambio': {'type': 'keyword'},
                'combustivel': {'type': 'keyword'},
                'destaques': {'type': 'text'},
                'descricao': {'type': 'text'},
                'texto_busca': {'type': 'text', 'analyzer': 'brazilian'},
                'vetor': {'type': 'knn_vector', 'dimension': dimension,
                          'method': {'name': 'hnsw', 'engine': 'lucene',
                                     'space_type': 'cosinesimil',
                                     'parameters': {'ef_construction': 100, 'm': 16}}},
            }},
        })
    actions = ({'_index': config.OPENSEARCH_INDEX, '_id': v['id'],
                '_source': {**v, 'texto_busca': text, 'vetor': vector}}
               for v, text, vector in zip(vehicles, texts, vectors))
    count, _ = helpers.bulk(client, actions, raise_on_error=True, request_timeout=180)
    client.indices.refresh(index=config.OPENSEARCH_INDEX)
    print(f'OK: {count} veículos indexados em {config.OPENSEARCH_INDEX}.')


if __name__ == '__main__':
    main()
