"""Configuração do laboratório (serviços locais)."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL', 'http://localhost:9200')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'lab_veiculos_rag')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'qwen2.5:1.5b')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'all-minilm')
TOP_K = int(os.getenv('TOP_K', '8'))
