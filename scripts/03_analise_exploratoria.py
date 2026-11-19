# %% [markdown]
# # Análise exploratória — evasão na Rede Federal do Nordeste
#
# Usa **somente a partição de treino** de `ef_2021_2023.csv`. Validação e
# teste ficam fechados até a avaliação, para que nada visto aqui influencie
# decisões medidas neles.
#
# Perguntas que esta análise responde:
# 1. Qual a qualidade dos atributos (ausentes, "Não declarada", categorias raras)?
# 2. Como a evasão varia com o curso, a oferta e o perfil do aluno?
# 3. Os atributos de perfil (sexo, cor/raça, renda) se confundem com o tipo de curso?
# 4. O IFPI difere do restante do Nordeste?
#
# As figuras são salvas em `figuras/` para reuso no artigo.

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEMENTE = 42
PASTA_FIG = Path("figuras")
PASTA_FIG.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})
COR = "#2b6cb0"      # destaque
CINZA = "#9aa5b1"

df_completo = pd.read_csv("dados/processados/ef_2021_2023.csv", dtype={"ano_base": int})
treino = df_completo[df_completo["particao"] == "treino"].copy()
TAXA_GERAL = treino["evadiu"].mean()
print(f"Treino: {len(treino):,} matrículas, evasão de {TAXA_GERAL:.1%}")

AUSENTES = {"Não informado", "Não declarada", "Não declarado", "nan", "Não Declarada"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}
treino["nivel"] = np.where(treino["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")


# %% [markdown]
# ## Funções auxiliares

# %%
def milhar(n):
    """12345 -> '12.345'."""
    return f"{int(n):,}".replace(",", ".")


def wilson(k, n, z=1.96):
    """Intervalo de confiança de 95% (Wilson) para uma proporção."""
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - meia, centro + meia


def taxa_por(df, coluna, minimo=1):
    """Evasão por categoria, com n e IC 95%."""
    t = df.groupby(coluna)["evadiu"].agg(n="size", evadidos="sum")
    t = t[t["n"] >= minimo].copy()
    t["taxa"] = t["evadidos"] / t["n"]
    t["ic_inf"], t["ic_sup"] = wilson(t["evadidos"], t["n"])
    return t


def grafico_taxa(df, coluna, titulo, ordem=None, minimo=30, arquivo=None, destaque=None):
    """Barras horizontais de evasão por categoria, com IC 95% e a taxa geral como referência."""
    t = taxa_por(df, coluna, minimo)
    t = t.reindex([o for o in ordem if o in t.index]) if ordem else t.sort_values("taxa")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.4))
    cores = [COR if (destaque and destaque(i)) else CINZA for i in t.index]
    ax.barh(t.index.astype(str), t["taxa"] * 100, color=cores,
            xerr=[(t["taxa"] - t["ic_inf"]) * 100, (t["ic_sup"] - t["taxa"]) * 100],
            error_kw={"ecolor": "#555", "capsize": 3, "lw": 1})
    ax.invert_yaxis()
    ax.axvline(TAXA_GERAL * 100, ls="--", lw=1, color="#555")
    for y, (taxa, sup, n) in enumerate(zip(t["taxa"], t["ic_sup"], t["n"])):
        ax.text(sup * 100 + 1.5, y, f"{taxa:.1%}  (n={milhar(n)})", va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"Taxa de evasão (%) · tracejado: geral {TAXA_GERAL:.1%} · barras: IC 95%")
    ax.set_title(titulo)
    fig.tight_layout()
    if arquivo:
        fig.savefig(PASTA_FIG / arquivo, bbox_inches="tight")
    plt.show()
    return t


def ordem_renda(categorias):
    """Ordena faixas de renda per capita pelo primeiro número; ausentes no fim."""
    def chave(c):
        if c in AUSENTES:
            return (1, 0.0)
        nums = re.findall(r"\d+(?:,\d+)?", c)
        return (0, float(nums[0].replace(",", ".")) if nums else 99.0)
    return sorted(categorias, key=chave)


# %% [markdown]
# ## 1. Qualidade dos dados
#
# Proporção de valores ausentes ou não declarados por atributo e edição.
# Atributos com muita ausência pedem cuidado: no modelo, "não declarado" pode
# virar categoria própria; na análise de viés, os grupos ficam menos confiáveis.

# %%
categoricas = [c for c in treino.columns if not pd.api.types.is_numeric_dtype(treino[c])
               and c not in ("particao", "nivel")]
ausencia = pd.DataFrame({
    ano: {c: g[c].isin(AUSENTES).mean() * 100 for c in categoricas}
    for ano, g in treino.groupby("ano_base")})
ausencia.loc["idade_ingresso"] = treino.groupby("ano_base")["idade_ingresso"].apply(
    lambda s: s.isna().mean() * 100)
ausencia = ausencia.round(2)
ausencia = ausencia[ausencia.max(axis=1) > 0].sort_values(ausencia.columns[-1], ascending=False)
print("Ausentes / não declarados (% das matrículas de treino):")
print(ausencia.to_string())

# %%
fig, ax = plt.subplots(figsize=(8, 0.45 * len(ausencia) + 1.4))
ausencia.plot.barh(ax=ax, color=["#c3cfe2", "#7f9cc7", COR], width=0.8)
ax.invert_yaxis()
ax.set_xlabel("% de matrículas sem a informação")
ax.set_title("Atributos com informação ausente ou não declarada")
ax.legend(title="Edição", frameon=False)
fig.tight_layout()
fig.savefig(PASTA_FIG / "01_ausentes.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Cardinalidade
#
# Quantas categorias cada atributo tem e quantas são raras (menos de 30
# matrículas no treino). Atributos com muitas categorias raras precisam de
# agrupamento ou de uma codificação que não decore cada valor.

# %%
cardinalidade = pd.DataFrame({
    "categorias": treino[categoricas].nunique(),
    "raras (<30)": [(treino[c].value_counts() < 30).sum() for c in categoricas],
    "% matrículas em raras": [
        treino[c].map(treino[c].value_counts()).lt(30).mean() * 100 for c in categoricas],
}).round(1).sort_values("categorias", ascending=False)
print(cardinalidade.to_string())

# %% [markdown]
# ## 2. Distribuição do rótulo

# %%
por_edicao = taxa_por(treino, "ano_base")
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(por_edicao.index.astype(str), por_edicao["taxa"] * 100,
       color=[COR if a == por_edicao["taxa"].idxmax() else CINZA for a in por_edicao.index],
       yerr=[(por_edicao["taxa"] - por_edicao["ic_inf"]) * 100,
             (por_edicao["ic_sup"] - por_edicao["taxa"]) * 100], capsize=4)
for x, (taxa, n) in enumerate(zip(por_edicao["taxa"], por_edicao["n"])):
    ax.text(x, taxa * 100 + 4, f"{taxa:.1%}\nn={milhar(n)}", ha="center", fontsize=8)
ax.set_ylim(0, 80)
ax.set_xlabel("Edição da Eficiência Acadêmica")
ax.set_ylabel("Taxa de evasão (%)")
ax.set_title("Evasão por edição (maior em destaque)")
fig.tight_layout()
fig.savefig(PASTA_FIG / "02_evasao_por_edicao.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Evasão por características do curso

# %%
grafico_taxa(treino, "tipo_de_curso", "Evasão por tipo de curso (graduação em destaque)",
             arquivo="03_tipo_de_curso.png", destaque=lambda c: c in GRADUACAO)

# %%
grafico_taxa(treino, "tipo_de_oferta", "Evasão por tipo de oferta",
             arquivo="04_tipo_de_oferta.png")

# %%
grafico_taxa(treino, "turno", "Evasão por turno", arquivo="05_turno.png")

# %%
grafico_taxa(treino, "modalidade_de_ensino", "Evasão por modalidade",
             arquivo="06_modalidade.png")

# %%
grafico_taxa(treino, "eixo_tecnologico", "Evasão por eixo tecnológico",
             arquivo="07_eixo.png")

# %%
# Duração prevista do curso, em faixas de 12 meses.
treino["faixa_duracao"] = pd.cut(treino["duracao_prevista_meses"],
                                 bins=[0, 12, 24, 36, 48, 60, 600],
                                 labels=["até 12", "13–24", "25–36", "37–48", "49–60", "mais de 60"])
grafico_taxa(treino, "faixa_duracao", "Evasão por duração prevista do curso (meses)",
             ordem=["até 12", "13–24", "25–36", "37–48", "49–60", "mais de 60"],
             arquivo="08_duracao.png")

# %% [markdown]
# ## 4. Evasão pelo perfil do aluno
#
# Estes são os atributos da análise de viés. Os intervalos de confiança mostram
# quais diferenças são estáveis e quais podem ser ruído de grupos pequenos.

# %%
grafico_taxa(treino, "sexo", "Evasão por sexo", arquivo="09_sexo.png")

# %%
grafico_taxa(treino, "cor_raca", "Evasão por cor/raça", arquivo="10_cor_raca.png",
             destaque=lambda c: c in AUSENTES)

# %%
renda = taxa_por(treino, "renda_familiar")
grafico_taxa(treino, "renda_familiar", "Evasão por renda familiar per capita (salários mínimos)",
             ordem=ordem_renda(renda.index), arquivo="11_renda.png",
             destaque=lambda c: c in AUSENTES)

# %%
treino["faixa_idade"] = pd.cut(treino["idade_ingresso"], bins=[0, 17, 20, 24, 29, 39, 120],
                               labels=["até 17", "18–20", "21–24", "25–29", "30–39", "40 ou mais"])
grafico_taxa(treino, "faixa_idade", "Evasão por idade no ingresso",
             ordem=["até 17", "18–20", "21–24", "25–29", "30–39", "40 ou mais"],
             arquivo="12_idade.png")

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.arange(10, 66, 2)
for rotulo, cor, nome in [(0, CINZA, "Concluiu"), (1, COR, "Evadiu")]:
    ax.hist(treino.loc[treino["evadiu"] == rotulo, "idade_ingresso"].dropna(),
            bins=bins, alpha=0.6, color=cor, label=nome, density=True)
ax.set_xlabel("Idade no ingresso (anos)")
ax.set_ylabel("Densidade")
ax.set_title("Distribuição da idade no ingresso por desfecho")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PASTA_FIG / "13_idade_hist.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Perfil e tipo de curso se confundem?
#
# Se um grupo se concentra na graduação, que evade mais, parte da diferença de
# evasão desse grupo vem do curso e não do perfil. Comparar dentro de cada
# nível (técnico e graduação) separa os dois efeitos.

# %%
for atributo in ["sexo", "cor_raca", "renda_familiar"]:
    t = (treino.groupby([atributo, "nivel"])["evadiu"]
               .agg(n="size", taxa="mean").unstack("nivel"))
    t = t[(t["n"] >= 30).all(axis=1)]
    composicao = pd.crosstab(treino[atributo], treino["nivel"], normalize="index") * 100
    saida = pd.DataFrame({
        "% em graduação": composicao["Graduação"].round(1),
        "evasão técnico (%)": (t["taxa"]["Técnico"] * 100).round(1),
        "evasão graduação (%)": (t["taxa"]["Graduação"] * 100).round(1),
        "n técnico": t["n"]["Técnico"], "n graduação": t["n"]["Graduação"],
    }).dropna()
    if atributo == "renda_familiar":
        saida = saida.reindex(ordem_renda(saida.index))
    print(f"\n=== {atributo} ===")
    print(saida.to_string())

# %% [markdown]
# ## 6. Instituições e o IFPI

# %%
inst = taxa_por(treino, "instituicao").sort_values("taxa", ascending=False)
fig, ax = plt.subplots(figsize=(8, 0.4 * len(inst) + 1.4))
eh_ifpi = inst.index == "Instituto Federal do Piauí"
ax.barh(inst.index, inst["taxa"] * 100, color=np.where(eh_ifpi, COR, CINZA),
        xerr=[(inst["taxa"] - inst["ic_inf"]) * 100, (inst["ic_sup"] - inst["taxa"]) * 100],
        error_kw={"ecolor": "#555", "capsize": 2, "lw": 1})
ax.axvline(TAXA_GERAL * 100, ls="--", lw=1, color="#555")
for y, (taxa, sup, n) in enumerate(zip(inst["taxa"], inst["ic_sup"], inst["n"])):
    ax.text(sup * 100 + 1.5, y, f"{taxa:.1%}  (n={milhar(n)})", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel(f"Taxa de evasão (%) · tracejado: geral {TAXA_GERAL:.1%} · barras: IC 95%")
ax.set_title("Evasão por instituição (IFPI em destaque)")
fig.tight_layout()
fig.savefig(PASTA_FIG / "14_instituicoes.png", bbox_inches="tight")
plt.show()

# %%
treino["grupo_ifpi"] = np.where(treino["instituicao"] == "Instituto Federal do Piauí",
                                "IFPI", "Resto do Nordeste")
comparacao = (treino.groupby(["nivel", "grupo_ifpi"])["evadiu"]
                    .agg(n="size", taxa="mean").unstack("grupo_ifpi"))
comparacao[("taxa", "IFPI")] *= 100
comparacao[("taxa", "Resto do Nordeste")] *= 100
print("IFPI x resto do Nordeste, por nível (taxa em %):")
print(comparacao.round(1).to_string())
print(f"\nMatrículas do IFPI no treino: {(treino['grupo_ifpi'] == 'IFPI').sum():,}")
print(f"Matrículas do IFPI no teste: "
      f"{(df_completo.query('particao == \"teste\"')['instituicao'] == 'Instituto Federal do Piauí').sum():,}")

# %% [markdown]
# ## 7. Resumo numérico para a documentação

# %%
resumo = {
    "matrículas de treino": len(treino),
    "evasão geral (%)": round(TAXA_GERAL * 100, 1),
    "% técnico": round((treino["nivel"] == "Técnico").mean() * 100, 1),
    "% cor/raça não declarada": round(treino["cor_raca"].isin(AUSENTES).mean() * 100, 1),
    "% renda não declarada": round(treino["renda_familiar"].isin(AUSENTES).mean() * 100, 1),
    "instituições": treino["instituicao"].nunique(),
    "cursos distintos": treino["nome_de_curso"].nunique(),
    "campi": treino["unidade_de_ensino"].nunique(),
}
for k, v in resumo.items():
    print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")
