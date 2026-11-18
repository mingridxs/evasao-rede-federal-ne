# %% [markdown]
# # Análise de viés do modelo principal
#
# Usa as previsões do **teste** salvas por `modelagem.py`
# (`resultados/predicoes_teste.csv`). Nenhum modelo é ajustado aqui.
#
# Para cada grupo de sexo, cor/raça, renda e idade, compara:
# - **Recall** (taxa de verdadeiros positivos): dos alunos do grupo que
#   evadiram, quantos o modelo marcou como risco. Recall menor = o grupo fica
#   mais sem apoio.
# - **Taxa de falsos positivos**: dos que concluíram, quantos foram marcados
#   como risco. Taxa maior = o grupo é mais rotulado sem motivo.
# - **Taxa de alerta**: proporção do grupo marcada como risco.
# - **Calibração**: probabilidade média prevista contra a evasão observada.
# - **AUC dentro do grupo**: se o modelo ordena bem os alunos daquele grupo.
#
# Como o tipo de curso pesa muito na evasão, tudo é repetido dentro do
# técnico e da graduação. Grupos com menos de 30 evadidos ou 30 concluintes
# não têm recall ou falsos positivos calculados: o intervalo seria largo
# demais para dizer algo.

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEMENTE = 42
MIN_CLASSE = 30
PASTA_FIG, PASTA_RES = Path("figuras"), Path("resultados")
PASTA_FIG.mkdir(exist_ok=True)
COR, CINZA, COR2 = "#2b6cb0", "#9aa5b1", "#dd6b20"
COR_NIVEL = {"Técnico": COR, "Graduação": COR2}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

AUSENTES = {"Não informado", "Não declarada", "Não declarado", "nan", "Não Declarada"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

p = pd.read_csv(PASTA_RES / "predicoes_teste.csv")
LIMIAR = float(p["limiar_principal"].iloc[0])
p["alerta"] = (p["prob_principal"] >= LIMIAR).astype(int)
p["nivel"] = np.where(p["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
p["faixa_idade"] = pd.cut(p["idade_ingresso"], bins=[0, 17, 20, 24, 29, 120],
                          labels=["até 17", "18–20", "21–24", "25–29", "30 ou mais"]).astype(str)
p["grupo_ifpi"] = np.where(p["instituicao"].str.contains("Piau", case=False), "IFPI",
                           "Resto do Nordeste")
print(f"Teste: {len(p):,} matrículas · limiar {LIMIAR:.2f} · evasão real {p['evadiu'].mean():.1%} "
      f"· taxa de alerta {p['alerta'].mean():.1%}")


# %% [markdown]
# ## Funções

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    q = k / n
    centro = (q + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(q * (1 - q) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - meia, centro + meia


def auc_ic(y, prob, n=300, semente=SEMENTE):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(semente)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], prob[i]))
    return roc_auc_score(y, prob), *np.percentile(aucs, [2.5, 97.5])


def metricas_por_grupo(df, coluna, prob="prob_principal"):
    linhas = []
    for grupo, g in df.groupby(coluna):
        pos, neg = g[g["evadiu"] == 1], g[g["evadiu"] == 0]
        l = {"grupo": grupo, "n": len(g), "evadidos": len(pos), "concluintes": len(neg),
             "evasão real": g["evadiu"].mean(), "prob. média": g[prob].mean(),
             "taxa de alerta": g["alerta"].mean()}
        l["erro de calibração"] = l["prob. média"] - l["evasão real"]
        if len(pos) >= MIN_CLASSE and len(neg) >= MIN_CLASSE:
            l["recall"] = pos["alerta"].mean()
            l["recall ic inf"], l["recall ic sup"] = wilson(pos["alerta"].sum(), len(pos))
            l["falsos positivos"] = neg["alerta"].mean()
            l["fp ic inf"], l["fp ic sup"] = wilson(neg["alerta"].sum(), len(neg))
            l["AUC"], l["AUC ic inf"], l["AUC ic sup"] = auc_ic(g["evadiu"].values, g[prob].values)
        linhas.append(l)
    colunas = ["n", "evadidos", "concluintes", "evasão real", "prob. média", "taxa de alerta",
               "erro de calibração", "recall", "recall ic inf", "recall ic sup",
               "falsos positivos", "fp ic inf", "fp ic sup", "AUC", "AUC ic inf", "AUC ic sup"]
    return pd.DataFrame(linhas).set_index("grupo").reindex(columns=colunas)


def ordem_renda(categorias):
    def chave(c):
        if c in AUSENTES:
            return (1, 0.0)
        nums = re.findall(r"\d+(?:,\d+)?", c)
        return (0, float(nums[0].replace(",", ".")) if nums else 99.0)
    return sorted(categorias, key=chave)


ORDENS = {
    "faixa_idade": ["até 17", "18–20", "21–24", "25–29", "30 ou mais"],
    "renda_familiar": None,   # calculada a partir dos dados
}


def ordenar(t, coluna):
    if coluna == "renda_familiar":
        return t.reindex(ordem_renda(t.index))
    if coluna in ORDENS and ORDENS[coluna]:
        return t.reindex([o for o in ORDENS[coluna] if o in t.index])
    return t.sort_values("n", ascending=False)


def resumo_disparidade(t, nome):
    """Maior diferença entre grupos com recall calculado."""
    v = t.dropna(subset=["recall"])
    v = v[~v.index.isin(AUSENTES)]
    if len(v) < 2:
        return None
    return {"atributo": nome, "grupos comparados": len(v),
            "diferença de recall": v["recall"].max() - v["recall"].min(),
            "menor recall": v["recall"].idxmin(),
            "diferença de falsos positivos": v["falsos positivos"].max() - v["falsos positivos"].min(),
            "maior taxa de falsos positivos": v["falsos positivos"].idxmax(),
            "maior erro de calibração (pontos)": (v["erro de calibração"].abs().max() * 100)}


def grafico_grupos(tabelas, coluna, titulo, arquivo):
    """Recall e falsos positivos por grupo, um ponto por nível de curso, com IC 95%."""
    grupos = list(dict.fromkeys(g for t in tabelas.values() for g in t.dropna(subset=["recall"]).index))
    if not grupos:
        return
    fig, eixos = plt.subplots(1, 2, figsize=(10, 0.5 * len(grupos) + 1.8), sharey=True)
    for ax, (met, inf, sup, rotulo) in zip(eixos, [
            ("recall", "recall ic inf", "recall ic sup", "Recall: evadidos marcados como risco (%)"),
            ("falsos positivos", "fp ic inf", "fp ic sup", "Concluintes marcados como risco (%)")]):
        for k, (nivel, t) in enumerate(tabelas.items()):
            t = t.reindex(grupos)
            y = np.arange(len(grupos)) + (k - 0.5) * 0.25
            ok = t[met].notna().values
            ax.errorbar(t[met].values[ok] * 100, y[ok],
                        xerr=[(t[met] - t[inf]).values[ok] * 100, (t[sup] - t[met]).values[ok] * 100],
                        fmt="o", color=COR_NIVEL[nivel], label=nivel, capsize=3, ms=5)
        ax.set_yticks(range(len(grupos)), grupos)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(rotulo)
        ax.grid(axis="x", alpha=0.3)
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, frameon=False, loc="upper right", ncol=2)
    fig.suptitle(titulo, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PASTA_FIG / arquivo, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 1. Visão geral por nível de curso

# %%
geral = metricas_por_grupo(p, "nivel")
print(geral.round(3).T.to_string())

# %% [markdown]
# ## 2. Grupos de perfil
#
# Para cada atributo: a tabela de todos os alunos, depois técnico e graduação
# separados, e o gráfico de recall e falsos positivos.

# %%
ATRIBUTOS = {"sexo": "Sexo", "cor_raca": "Cor/raça", "renda_familiar": "Renda familiar per capita",
             "faixa_idade": "Idade no ingresso"}
COLS_TABELA = ["n", "evasão real", "prob. média", "erro de calibração", "taxa de alerta",
               "recall", "falsos positivos", "AUC"]

tabelas_salvas, disparidades = [], []
for i, (coluna, nome) in enumerate(ATRIBUTOS.items(), start=30):
    todos = ordenar(metricas_por_grupo(p, coluna), coluna)
    por_nivel = {nivel: ordenar(metricas_por_grupo(g, coluna), coluna)
                 for nivel, g in p.groupby("nivel")}
    print(f"\n========== {nome} ==========")
    print("Todos:")
    print(todos[COLS_TABELA].round(3).to_string())
    for nivel, t in por_nivel.items():
        print(f"\n{nivel}:")
        print(t[COLS_TABELA].round(3).to_string())
        d = resumo_disparidade(t, f"{nome} · {nivel}")
        if d:
            disparidades.append(d)
    d = resumo_disparidade(todos, f"{nome} · todos")
    if d:
        disparidades.append(d)
    for escopo, t in [("todos", todos), *por_nivel.items()]:
        tabelas_salvas.append(t.assign(atributo=coluna, escopo=escopo))
    grafico_grupos(por_nivel, coluna, f"{nome}: recall e falsos positivos por nível de curso",
                   f"{i}_vies_{coluna}.png")

pd.concat(tabelas_salvas).to_csv(PASTA_RES / "vies_por_grupo.csv")

# %% [markdown]
# ## 3. Calibração por grupo
#
# Pontos sobre a diagonal = a probabilidade média prevista para o grupo bate
# com a evasão que ele teve. Abaixo da diagonal = o modelo superestima o
# risco do grupo; acima = subestima.

# %%
pontos = pd.concat([
    ordenar(metricas_por_grupo(g, c), c).assign(atributo=ATRIBUTOS[c], nivel=nivel)
    for c in ATRIBUTOS for nivel, g in p.groupby("nivel")])
pontos = pontos[pontos["n"] >= 100]
fig, ax = plt.subplots(figsize=(6, 6))
for nivel, cor in COR_NIVEL.items():
    s = pontos[pontos["nivel"] == nivel]
    ax.scatter(s["prob. média"] * 100, s["evasão real"] * 100, s=np.clip(s["n"] / 15, 15, 200),
               color=cor, alpha=0.7, label=nivel)
pior = pontos.loc[pontos["erro de calibração"].abs().nlargest(4).index.unique()]
for g, l in pior.iterrows():
    ax.annotate(f"{l['atributo']}: {g} ({l['nivel'][:4]}.)", (l["prob. média"] * 100, l["evasão real"] * 100),
                textcoords="offset points", xytext=(6, -10), fontsize=8)
ax.plot([0, 100], [0, 100], ":", color="#555")
ax.set_xlim(20, 100)
ax.set_ylim(20, 100)
ax.set_xlabel("Probabilidade média prevista (%)")
ax.set_ylabel("Evasão observada (%)")
ax.set_title("Calibração por grupo (tamanho = n; grupos com n ≥ 100)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(PASTA_FIG / "34_calibracao_grupos.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. IFPI

# %%
ifpi = pd.concat({nivel: metricas_por_grupo(g, "grupo_ifpi") for nivel, g in p.groupby("nivel")})
print(ifpi[COLS_TABELA + ["recall ic inf", "recall ic sup"]].round(3).to_string())

# %% [markdown]
# ## 5. Resumo das disparidades
#
# Para cada atributo e escopo, a maior diferença de recall e de falsos
# positivos entre grupos declarados (o "não declarado" fica de fora da
# comparação, porque não é um grupo de pessoas e sim uma falta de registro).

# %%
resumo = pd.DataFrame(disparidades)
for c in ["diferença de recall", "diferença de falsos positivos"]:
    resumo[c] = (resumo[c] * 100).round(1)
resumo["maior erro de calibração (pontos)"] = resumo["maior erro de calibração (pontos)"].round(1)
resumo.to_csv(PASTA_RES / "vies_resumo.csv", index=False)
print(resumo.to_string(index=False))
