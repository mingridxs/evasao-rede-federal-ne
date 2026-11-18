"""
Pré-processamento dos microdados de matrículas da Plataforma Nilo Peçanha (PNP)
para o projeto de previsão de evasão.

Uso, na raiz do repositório:
    python scripts/01_preprocessa_pnp.py dados/brutos/microdados_eficiencia_academica_2023.csv \
        --ano 2023 --regiao Nordeste --saida dados/processados/ef_2023.csv

Passe um arquivo por ano-base ({2021: "...", 2022: "...", 2023: "..."}) para
incluir quem evadiu em anos anteriores e poder dividir treino e teste por
ano de ingresso.
"""
import re
import unicodedata

import pandas as pd

# Atributos conhecidos no ingresso (nomes já normalizados).
ATRIBUTOS = [
    "instituicao", "unidade_de_ensino", "uf", "regiao",
    "nome_de_curso", "tipo_de_curso", "tipo_de_oferta", "modalidade_de_ensino",
    "eixo_tecnologico", "subeixo_tecnologico", "turno", "fonte_de_financiamento",
    "carga_horaria",
    "sexo", "cor_raca", "renda_familiar", "idade",
]
DATAS = ["data_de_inicio_do_ciclo", "data_de_fim_previsto_do_ciclo",
         "data_de_ocorrencia_da_matricula"]
# No arquivo de 2023 a coluna 'Região' vem preenchida com o nome da instituição,
# então a região é derivada da UF.
REGIAO_POR_UF = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}
# Cursos regulares de nível técnico e de graduação. Ficam fora, por padrão,
# qualificação (FIC), pós-graduação e ensino médio, em que a evasão é outro fenômeno.
TIPOS_PADRAO = ["Técnico", "Tecnologia", "Licenciatura", "Bacharelado"]
ROTULO = "categoria_da_situacao"
ID = "codigo_da_matricula"

# Ficam de fora de propósito:
# - situacao_de_matricula e mes_de_ocorrencia_da_situacao: revelam o rótulo (vazamento);
# - matricula_atendida, numero_de_registros, fator_esforco_curso, vagas_*, total_de_inscritos:
#   indicadores de gestão, não características do aluno;
# - códigos (co_inst, cod_unidade, SISTEC, município): redundantes com os nomes;
# - faixa_etaria: redundante com idade.


def normaliza(nome: str) -> str:
    """'Cor / Raça' -> 'cor_raca'."""
    sem_acento = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


def le_csv(caminho: str) -> pd.DataFrame:
    """Lê o CSV detectando codificação e separador."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(caminho, encoding=enc) as f:
                cabecalho = f.readline()
            sep = max([";", "\t", ","], key=cabecalho.count)
            return pd.read_csv(caminho, sep=sep, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não consegui ler {caminho}")


def sem_acento(serie: pd.Series) -> pd.Series:
    """Minúsculas e sem acento, para comparar 'Piauí' com 'piaui'."""
    return (serie.fillna("").str.normalize("NFKD")
            .str.encode("ascii", "ignore").str.decode("ascii").str.lower())


def preprocessa(arquivos: dict, regiao: str | None = None, uf: str | None = None,
                instituicao: str | None = None, tipos_de_curso: list | None = TIPOS_PADRAO,
                so_ciclos_encerrados: bool = True) -> pd.DataFrame:
    """
    arquivos: {ano_base: caminho_do_csv}
    regiao / uf / instituicao: recortes opcionais e combináveis
        (ex.: regiao="Nordeste", uf="PI", instituicao="Piauí")
    tipos_de_curso: lista de tipos mantidos; None mantém todos
    so_ciclos_encerrados: mantém só matrículas cujo fim previsto do ciclo já passou
        no último ano-base carregado, para não comparar quem evadiu cedo com quem
        ainda nem teria tido tempo de concluir.
    """
    partes = []
    for ano_base, caminho in arquivos.items():
        df = le_csv(caminho)
        df.columns = [normaliza(c) for c in df.columns]
        # Usa o ano-base gravado no próprio arquivo (coluna 'Ano'), se existir.
        ano_arquivo = pd.to_numeric(df.get("ano"), errors="coerce") if "ano" in df.columns else None
        if ano_arquivo is not None and ano_arquivo.notna().all():
            anos = sorted(ano_arquivo.astype(int).unique())
            if anos != [int(ano_base)]:
                print(f"AVISO - {caminho}: informado {ano_base}, coluna 'Ano' traz {[int(a) for a in anos]}; "
                      "usando o valor do arquivo")
            df["ano_base"] = ano_arquivo.astype(int)
        else:
            df["ano_base"] = int(ano_base)
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    print(f"Linhas lidas: {len(df):,}")

    # Confere se as colunas esperadas existem.
    esperadas = ATRIBUTOS + DATAS + [ROTULO]
    faltando = [c for c in esperadas if c not in df.columns]
    if ROTULO in faltando:
        raise KeyError(f"Coluna de rótulo ausente. Colunas do arquivo: {list(df.columns)}")
    if faltando:
        print(f"AVISO - colunas não encontradas (ignoradas): {faltando}")
    atributos = [c for c in ATRIBUTOS if c in df.columns]
    datas = [c for c in DATAS if c in df.columns]

    # Limpa espaços e padroniza vazios.
    for c in atributos + [ROTULO]:
        df[c] = df[c].str.strip().replace({"": pd.NA, "-": pd.NA})

    # Recortes.
    if "uf" in df.columns:
        df["uf"] = df["uf"].str.upper()
        df["regiao"] = df["uf"].map(REGIAO_POR_UF)
    if regiao:
        df = df[sem_acento(df["regiao"]).map(normaliza) == normaliza(regiao)]
    if uf and "uf" in df.columns:
        df = df[df["uf"] == uf.upper()]
    if instituicao and "instituicao" in df.columns:
        alvo = normaliza(instituicao).replace("_", " ")
        df = df[sem_acento(df["instituicao"]).str.contains(alvo, regex=False)]
    if tipos_de_curso and "tipo_de_curso" in df.columns:
        df = df[df["tipo_de_curso"].isin(tipos_de_curso)]
    print(f"Após recorte: {len(df):,}")

    # Duplicatas: mesma matrícula no mesmo ano-base.
    if ID in df.columns:
        df = df.drop_duplicates(subset=[ID, "ano_base"])

    # Transferência interna (mudou de curso na mesma instituição) não é evasão.
    # (A PNP conta como evasão; aqui é uma decisão do trabalho, registrada na metodologia.)
    if "situacao_de_matricula" in df.columns:
        df = df[~sem_acento(df["situacao_de_matricula"]).str.contains("interna", regex=False)]

    # Rótulo: só matrículas com desfecho. 'Em Curso' sai (no arquivo de Eficiência
    # Acadêmica, são alunos ainda ativos depois do fim previsto do ciclo: retidos).
    # 'Concluídos' (Concluída) e 'Concluintes' (Integralizada) contam como não evasão.
    cat = df[ROTULO].str.lower().fillna("")
    df = df[cat.str.startswith(("evadid", "conclu"))].copy()
    df["evadiu"] = df[ROTULO].str.lower().str.startswith("evadid").astype(int)
    print(f"Com desfecho (evadido ou concluinte): {len(df):,}")

    # Com vários anos-base, a mesma matrícula pode aparecer mais de uma vez:
    # fica o registro mais recente.
    if ID in df.columns and df["ano_base"].nunique() > 1:
        repetidas = df[ID].duplicated().sum()
        df = df.sort_values("ano_base").drop_duplicates(subset=[ID], keep="last")
        print(f"Matrículas repetidas entre anos (removidas): {repetidas:,}")

    # Datas e atributos derivados.
    for c in datas:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    if "data_de_inicio_do_ciclo" in df.columns:
        df["ano_ingresso"] = df["data_de_inicio_do_ciclo"].dt.year
        df["mes_ingresso"] = df["data_de_inicio_do_ciclo"].dt.month
    if {"data_de_inicio_do_ciclo", "data_de_fim_previsto_do_ciclo"} <= set(df.columns):
        dias = (df["data_de_fim_previsto_do_ciclo"] - df["data_de_inicio_do_ciclo"]).dt.days
        df["duracao_prevista_meses"] = (dias / 30.44).round()
        if so_ciclos_encerrados:
            fim_ano = pd.Timestamp(year=int(df["ano_base"].max()), month=12, day=31)
            df = df[df["data_de_fim_previsto_do_ciclo"] <= fim_ano]
            print(f"Com ciclo já encerrado: {len(df):,}")

    # Demanda do curso no ano-base: inscritos e vagas da fase inicial.
    # (Segundo o Guia PNP, os dois se referem ao ano-base da edição, não ao
    # ano de ingresso do aluno.)
    derivados_demanda = []
    if "total_de_inscritos" in df.columns:
        cols_vagas = [c for c in df.columns if c.startswith("vagas_")]
        df["inscritos"] = pd.to_numeric(df["total_de_inscritos"], errors="coerce")
        df["vagas"] = df[cols_vagas].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        df["inscritos_por_vaga"] = df["inscritos"] / df["vagas"].where(df["vagas"] > 0)
        derivados_demanda = ["inscritos", "vagas", "inscritos_por_vaga"]

    # Numéricos.
    for c in ("idade", "carga_horaria"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    # Segundo as instruções dos microdados, 'idade' é a idade em 31/12 do ano-base.
    # Para usar só o que se sabe no ingresso, recua-se até o ano de ingresso.
    if {"idade", "ano_ingresso"} <= set(df.columns):
        df["idade_ingresso"] = df["idade"] - (df["ano_base"] - df["ano_ingresso"])
        df.loc[~df["idade_ingresso"].between(10, 90), "idade_ingresso"] = pd.NA
        atributos = [c for c in atributos if c != "idade"]
        derivados_idade = ["idade_ingresso"]
    else:
        derivados_idade = []

    # Categóricas: 'Não declarada' fica como categoria própria (a ausência é informativa).
    categoricas = [c for c in atributos if c != "carga_horaria"]
    df[categoricas] = df[categoricas].fillna("Não informado")

    derivados = [c for c in ("ano_ingresso", "mes_ingresso", "duracao_prevista_meses")
                 if c in df.columns] + derivados_idade + derivados_demanda
    final = df[["ano_base"] + atributos + derivados + ["evadiu"]].reset_index(drop=True)

    print(f"\nBase final: {final.shape[0]:,} linhas x {final.shape[1]} colunas")
    print(f"Taxa de evasão: {final['evadiu'].mean():.1%}")
    if "tipo_de_curso" in final.columns:
        print("\nPor tipo de curso (matrículas e taxa de evasão):")
        print(final.groupby("tipo_de_curso")["evadiu"].agg(["size", "mean"])
              .sort_values("size", ascending=False).round(3).to_string())
    print("\nValores ausentes por coluna:")
    print(final.isna().sum()[lambda s: s > 0].to_string() or "  nenhum")
    return final


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Limpa os microdados de matrículas da PNP")
    ap.add_argument("csv", nargs="+",
                    help="um arquivo, ou vários no formato ANO=arquivo.csv")
    ap.add_argument("--ano", type=int, default=2023,
                    help="ano-base, quando se passa um arquivo só")
    ap.add_argument("--saida", default="dados/processados/pnp_limpo.csv", help="nome do arquivo de saída")
    ap.add_argument("--todos-os-tipos", action="store_true",
                    help="inclui FIC, pós-graduação e ensino médio")
    ap.add_argument("--regiao")
    ap.add_argument("--uf")
    ap.add_argument("--instituicao")
    a = ap.parse_args()
    arquivos = {}
    for item in a.csv:
        ano, _, caminho = item.rpartition("=")
        arquivos[int(ano) if ano else a.ano] = caminho
    saida = preprocessa(arquivos, regiao=a.regiao, uf=a.uf, instituicao=a.instituicao,
                        tipos_de_curso=None if a.todos_os_tipos else TIPOS_PADRAO)
    saida.to_csv(a.saida, index=False)
    print(f"Salvo em {a.saida}")
