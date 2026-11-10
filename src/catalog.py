"""Filtros determinísticos: requisitos objetivos jamais ficam a cargo do LLM."""
import re
import unicodedata
from typing import Any

BRANDS = ('fiat', 'toyota', 'honda', 'hyundai', 'volkswagen', 'chevrolet',
          'renault', 'nissan', 'jeep', 'ford', 'peugeot', 'citroen')


def normalize(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(c))


def extract_filters(question: str) -> dict[str, Any]:
    """Extrai um subconjunto deliberadamente simples da linguagem natural.

    Suporta teto/preço mínimo, ano mínimo, câmbio, carroceria, combustível e marca.
    Para uma aplicação real: filtros explícitos de UI ou parser estruturado validado.
    """
    q = normalize(question)
    filters: dict[str, Any] = {}
    for label in ('suv', 'sedan', 'hatch', 'picape'):
        if re.search(rf'\b{label}\b', q):
            filters['carroceria'] = label
            break
    if 'automatic' in q or 'cvt' in q:
        filters['cambio'] = 'automatico'
    elif 'manual' in q:
        filters['cambio'] = 'manual'
    for name in ('diesel', 'flex', 'eletrico', 'hibrido', 'gasolina'):
        if re.search(rf'\b{name}\b', q):
            filters['combustivel'] = name
            break
    for brand in BRANDS:
        if re.search(rf'\b{brand}\b', q):
            filters['marca'] = brand
            break

    value_rx = r'(\d+(?:[.\s]\d{3})*)(?:,\d{1,2})?\s*(milhoes|milhao|mil|k)?'
    upper = re.search(r'\b(?:ate|abaixo de|menos de|no maximo)\s*(?:r\$\s*)?' + value_rx, q)
    lower = re.search(r'\b(?:acima de|mais de|a partir de)\s*(?:r\$\s*)?' + value_rx, q)

    def to_brl(match: re.Match) -> int:
        value = int(re.sub(r'\D', '', match.group(1)))
        unit = match.group(2)
        return value * (1_000_000 if unit in ('milhao', 'milhoes') else 1000 if unit in ('mil', 'k') else 1)

    for key, match in [('preco_max', upper), ('preco_min', lower)]:
        if match is None or re.match(r'\s*km\b', q[match.end():]):
            continue
        raw_number = int(re.sub(r'\D', '', match.group(1)))
        # Evita confundir "a partir de 2021" (ano) com preço mínimo.
        if 1900 <= raw_number <= 2099 and not match.group(2) and 'r$' not in match.group(0):
            continue
        filters[key] = to_brl(match)
    km = re.search(r'\b(?:menos de|ate|abaixo de)\s+(\d+)\s*(mil|k)?\s*km\b', q)
    if km:
        filters['km_max'] = int(km.group(1)) * (1000 if km.group(2) else 1)
    year = re.search(r'\b(?:a partir de|desde|de)\s*((?:19|20)\d{2})\s*(?:em diante|ou mais novo)?\b', q)
    if year:
        filters['ano_min'] = int(year.group(1))
    return filters


def opensearch_filters(filters: dict[str, Any]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for key in ('carroceria', 'cambio', 'combustivel', 'marca'):
        if key in filters:
            field = 'marca.raw' if key == 'marca' else key
            clauses.append({'term': {field: filters[key]}})
    price: dict[str, Any] = {}
    if 'preco_max' in filters:
        price['lte'] = filters['preco_max']
    if 'preco_min' in filters:
        price['gte'] = filters['preco_min']
    if price:
        clauses.append({'range': {'preco': price}})
    if 'ano_min' in filters:
        clauses.append({'range': {'ano': {'gte': filters['ano_min']}}})
    if 'km_max' in filters:
        clauses.append({'range': {'km': {'lte': filters['km_max']}}})
    return clauses


def build_search_text(vehicle: dict[str, Any]) -> str:
    return ('{marca} {modelo} {versao}. Carroceria {carroceria}. '
            'Câmbio {cambio}. Combustível {combustivel}. Ano {ano}. '
            'Destaques: {destaques}. {descricao}').format(**vehicle)
