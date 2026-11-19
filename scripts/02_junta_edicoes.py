# %% [markdown]
# # Junção das edições e divisão em treino, validação e teste
#
# Junta as bases limpas de Eficiência Acadêmica (edições 2021, 2022 e 2023,
# geradas por `preprocessa_pnp.py`) e faz a divisão aleatória estratificada
# 70% treino / 20% validação / 10% teste, estratificando por edição e rótulo.
#
# Saída: `ef_2021_2023.csv`, com a coluna `particao` indicando onde cada
# matrícula caiu. Guardar a partição no arquivo garante que todo o notebook
# (e quem reproduzir o trabalho) use exatamente a mesma divisão.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

SEMENTE = 42
EDICOES = [2021, 2022, 2023]
PASTA_DADOS = "dados/processados/"
ARQUIVOS = {ano: f"{PASTA_DADOS}ef_{ano}.csv" for ano in EDICOES}

# %% [markdown]
# ## 1. Carregar e conferir as edições

# %%
bases = {ano: pd.read_csv(caminho) for ano, caminho in ARQUIVOS.items()}

colunas = {ano: list(df.columns) for ano, df in bases.items()}
referencia = colunas[EDICOES[0]]
for ano, cols in colunas.items():
    if cols != referencia:
        faltam = set(referencia) - set(cols)
        sobram = set(cols) - set(referencia)
        raise ValueError(f"Edição {ano} tem colunas diferentes. Faltam: {faltam}. Sobram: {sobram}")
print(f"As {len(EDICOES)} edições têm as mesmas {len(referencia)} colunas.")

for ano, df in bases.items():
    anos_no_arquivo = sorted(df["ano_base"].unique())
    if anos_no_arquivo != [ano]:
        print(f"AVISO - ef_{ano}.csv traz ano_base {anos_no_arquivo}")

# %% [markdown]
# ## 2. Juntar

# %%
df = pd.concat(bases.values(), ignore_index=True)

# As categóricas viram texto em todas as edições, para não misturar tipos
# (ex.: um campo lido como número numa edição e como texto em outra).
numericas = ["carga_horaria", "ano_ingresso", "mes_ingresso",
             "duracao_prevista_meses", "idade_ingresso", "ano_base", "evadiu",
             "inscritos", "vagas", "inscritos_por_vaga"]
numericas = [c for c in numericas if c in df.columns]
categoricas = [c for c in df.columns if c not in numericas]
df[categoricas] = df[categoricas].astype(str)

# %% [markdown]
# ## 2b. Padronizar nomes que mudaram entre edições
#
# A edição 2023 traz o nome completo das instituições ("Instituto Federal do
# Piauí"); as de 2021 e 2022 trazem a sigla ("IFPI"), e as escolas técnicas
# vinculadas a universidades aparecem com siglas escritas de formas diferentes
# ("UFPI - CTF", "UFPI-CTF"). Sem padronizar, o modelo trataria a mesma
# instituição como duas, uma para 2021–2022 e outra para 2023.
# Padroniza também diferenças só de maiúsculas e minúsculas no nome do curso
# e do campus ("Informática Para Internet" e "Informática para Internet").

# %%
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
antes = df["instituicao"].nunique()
sigla = df["instituicao"].str.strip().str.replace(r"\s*-\s*", "-", regex=True).str.upper()
df["instituicao"] = sigla.map({k.upper(): v for k, v in NOME_INSTITUICAO.items()}).fillna(df["instituicao"].str.strip())
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
    print(f"{coluna}: {antes} grafias antes, {df[coluna].nunique()} depois.")

resumo = (df.groupby("ano_base")["evadiu"]
            .agg(matriculas="size", taxa_evasao="mean")
            .assign(taxa_evasao=lambda t: (t["taxa_evasao"] * 100).round(1)))
print(resumo.to_string())
print(f"\nTotal: {len(df):,} matrículas, evasão de {df['evadiu'].mean():.1%}")

# %% [markdown]
# ## 3. Dividir 70 / 20 / 10
#
# Estratifica pela combinação edição × rótulo, para que as três partes tenham
# a mesma mistura de anos e a mesma taxa de evasão. Codificação, normalização
# e imputação devem ser ajustadas **só no treino**, depois desta etapa.

# %%
estrato = df["ano_base"].astype(str) + "_" + df["evadiu"].astype(str)
idx_treino, idx_resto = train_test_split(
    df.index, test_size=0.30, stratify=estrato, random_state=SEMENTE)

idx_validacao, idx_teste = train_test_split(
    idx_resto, test_size=1/3, stratify=estrato.loc[idx_resto], random_state=SEMENTE)

df["particao"] = "treino"
df.loc[idx_validacao, "particao"] = "validacao"
df.loc[idx_teste, "particao"] = "teste"

conferencia = (df.groupby("particao")
                 .agg(matriculas=("evadiu", "size"), taxa_evasao=("evadiu", "mean"))
                 .assign(proporcao=lambda t: (t["matriculas"] / len(df) * 100).round(1),
                         taxa_evasao=lambda t: (t["taxa_evasao"] * 100).round(1))
                 .loc[["treino", "validacao", "teste"]])
print(conferencia.to_string())

print("\nEdições por partição (%):")
print((pd.crosstab(df["particao"], df["ano_base"], normalize="index") * 100)
      .round(1).loc[["treino", "validacao", "teste"]].to_string())

# %% [markdown]
# ## 4. Salvar

# %%
df.to_csv(PASTA_DADOS + "ef_2021_2023.csv", index=False)
print("Salvo em dados/processados/ef_2021_2023.csv")

# Para usar depois:
# df = pd.read_csv("dados/processados/ef_2021_2023.csv")
# treino    = df[df["particao"] == "treino"]
# validacao = df[df["particao"] == "validacao"]
# teste     = df[df["particao"] == "teste"]
