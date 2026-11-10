from src.catalog import extract_filters, opensearch_filters


def test_preco_suv_automatico():
    got = extract_filters('Quero um SUV automático até R$ 100 mil para minha família')
    assert got == {'carroceria': 'suv', 'cambio': 'automatico', 'preco_max': 100000}


def test_preco_com_ponto_marca_e_ano():
    got = extract_filters('Toyota a partir de 2021 até R$ 120.000')
    assert got == {'marca': 'toyota', 'preco_max': 120000, 'ano_min': 2021}


def test_combustivel_faixa():
    got = extract_filters('picape diesel acima de 80000 e até 150 mil')
    assert got['carroceria'] == 'picape'
    assert got['combustivel'] == 'diesel'
    assert got['preco_min'] == 80000
    assert got['preco_max'] == 150000
    assert {'range': {'preco': {'lte': 150000, 'gte': 80000}}} in opensearch_filters(got)


def test_limite_quilometragem_nao_contamina_preco():
    got = extract_filters('SUV automático até R$ 100 mil com menos de 50 mil km')
    assert got['preco_max'] == 100000
    assert got['km_max'] == 50000
