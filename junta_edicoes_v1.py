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
ARQUIVOS = {ano: f"ef_{ano}.csv" for ano in EDICOES}

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
             "duracao_prevista_meses", "idade_ingresso", "ano_base", "evadiu"]
numericas = [c for c in numericas if c in df.columns]
categoricas = [c for c in df.columns if c not in numericas]
df[categoricas] = df[categoricas].astype(str)

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
df.to_csv("ef_2021_2023.csv", index=False)
print("Salvo em ef_2021_2023.csv")

# Para usar depois:
# df = pd.read_csv("ef_2021_2023.csv")
# treino    = df[df["particao"] == "treino"]
# validacao = df[df["particao"] == "validacao"]
# teste     = df[df["particao"] == "teste"]
