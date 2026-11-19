"""
Padroniza nomes que mudam entre edições da Plataforma Nilo Peçanha.

A edição 2023 traz o nome completo das instituições ("Instituto Federal do
Piauí"); as de 2021 e 2022 trazem a sigla ("IFPI"), e as escolas técnicas
vinculadas a universidades aparecem com siglas escritas de formas diferentes
("UFPI - CTF", "UFPI-CTF"). Também une grafias do nome do curso e do campus que
só diferem em maiúsculas e minúsculas. Usado pelos scripts 02 e 15.
"""
import pandas as pd

NOME_INSTITUICAO = {
    "IF BAIANO": "Instituto Federal Baiano",
    "IF SERTÃO-PE": "Instituto Federal do Sertão Pernambucano",
    "IFAL": "Instituto Federal de Alagoas",
    "IFBA": "Instituto Federal da Bahia",
    "IFCE": "Instituto Federal do Ceará",
    "IFMA": "Instituto Federal do Maranhão",
    "IFPB": "Instituto Federal da Paraíba",
    "IFPE": "Instituto Federal de Pernambuco",
    "IFPI": "Instituto Federal do Piauí",
    "IFRN": "Instituto Federal do Rio Grande do Norte",
    "IFS": "Instituto Federal de Sergipe",
    "ESUFRN": "Escola de Saúde da UFRN",
    "UFAL-ETA": "Escola Técnica de Artes da UFAL",
    "UFCG-ETSC": "Escola Técnica de Saúde de Cajazeiras da UFCG",
    "UFMA-COLUN": "Colégio Universitário da UFMA",
    "UFPB-CAVN": "Colégio Agrícola Vidal de Negreiros da UFPB",
    "UFPB-ETS": "Escola Técnica de Saúde da UFPB",
    "UFPB-ESTES": "Escola Técnica de Saúde da UFPB",
    "UFPI-CABJ": "Colégio Técnico de Bom Jesus da UFPI",   # antigo Colégio Agrícola de Bom Jesus
    "UFPI-CTBJ": "Colégio Técnico de Bom Jesus da UFPI",
    "UFPI-CAT": "Colégio Técnico de Teresina da UFPI",     # antigo Colégio Agrícola de Teresina
    "UFPI-CTF": "Colégio Técnico de Floriano da UFPI",
    "UFRN-EAJ": "Escola Agrícola de Jundiaí da UFRN",
    "UFRN-MÚSICA": "Escola de Música da UFRN",
    "UFRN-MUSICA": "Escola de Música da UFRN",
    "UFRPE-CODAI": "Colégio Agrícola Dom Agostinho Ikas da UFRPE",
}


def padroniza_nomes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    antes = df["instituicao"].nunique()
    sigla = df["instituicao"].str.strip().str.replace(r"\s*-\s*", "-", regex=True).str.upper()
    df["instituicao"] = sigla.map({k.upper(): v for k, v in NOME_INSTITUICAO.items()}).fillna(df["instituicao"].str.strip())
    if verbose:
        print(f"Instituições: {antes} nomes antes, {df['instituicao'].nunique()} depois de padronizar.")
        por_edicao = pd.crosstab(df["instituicao"], df["ano_base"])
        incompletas = por_edicao[(por_edicao == 0).any(axis=1)]
        if len(incompletas):
            print("AVISO - instituições que não aparecem em todas as edições (confira se é outro nome a padronizar):")
            print(incompletas.to_string())
    for coluna in ["nome_de_curso", "unidade_de_ensino"]:
        chave = df[coluna].str.strip().str.casefold()
        mais_comum = df.groupby(chave)[coluna].agg(lambda s: s.value_counts().index[0])
        antes = df[coluna].nunique()
        df[coluna] = chave.map(mais_comum)
        if verbose:
            print(f"{coluna}: {antes} grafias antes, {df[coluna].nunique()} depois.")
    return df
