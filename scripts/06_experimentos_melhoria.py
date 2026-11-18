# %% [markdown]
# # Experimentos de melhoria do modelo principal
#
# Três perguntas, todas respondidas na **validação**; o teste é usado uma
# única vez, na seção 5, só para o que for escolhido aqui.
#
# 1. **Novos atributos** ajudam?
#    - *idade relativa*: idade no ingresso menos a idade mediana de quem entra
#      no mesmo curso, campus e tipo de oferta (calculada só no treino);
#    - *demanda*: inscritos por vaga do curso no ano-base (precisa das colunas
#      `inscritos`, `vagas` e `inscritos_por_vaga`, geradas pela versão atual
#      de `preprocessa_pnp.py`).
# 2. **Limiar por nível de curso** (técnico e graduação) melhora as decisões?
# 3. **Curva de aprendizado:** mais dados ajudariam?
#
# Hiperparâmetros: os do modelo principal escolhidos em `modelagem.py`
# (a busca mostrou que eles quase não importam).

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEMENTE = 42
PASTA_FIG, PASTA_RES = Path("figuras"), Path("resultados")
COR, CINZA, COR2 = "#2b6cb0", "#9aa5b1", "#dd6b20"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

CAT = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso",
       "tipo_de_oferta", "modalidade_de_ensino", "eixo_tecnologico", "subeixo_tecnologico",
       "turno", "fonte_de_financiamento", "sexo"]
NUM = ["carga_horaria", "duracao_prevista_meses", "mes_ingresso", "idade_ingresso"]
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

anterior = joblib.load(PASTA_RES / "modelo_principal.joblib")
PARAMS = anterior["hiperparametros"]
print("Hiperparâmetros do modelo principal:", PARAMS)

df = pd.read_csv("dados/processados/ef_2021_2023.csv", dtype={c: str for c in CAT})
treino = df[df["particao"] == "treino"].reset_index(drop=True)
validacao = df[df["particao"] == "validacao"].reset_index(drop=True)
teste = df[df["particao"] == "teste"].reset_index(drop=True)
for d in (treino, validacao, teste):
    d["nivel"] = np.where(d["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
y_tr, y_va, y_te = treino["evadiu"].values, validacao["evadiu"].values, teste["evadiu"].values
TEM_DEMANDA = "inscritos_por_vaga" in df.columns
print(f"Colunas de demanda disponíveis: {TEM_DEMANDA}")

# %% [markdown]
# ## 1. Novos atributos
#
# ### 1.1 Idade relativa ao curso
#
# A mediana vem só do treino. Para cursos que não aparecem no treino, cai
# para a mediana do nome de curso e, por fim, do tipo de curso.

# %%
NIVEIS_MEDIANA = [["unidade_de_ensino", "nome_de_curso", "tipo_de_oferta"],
                  ["nome_de_curso", "tipo_de_oferta"],
                  ["tipo_de_curso", "tipo_de_oferta"],
                  ["tipo_de_curso"]]
medianas = [treino.groupby(chaves)["idade_ingresso"].median().rename("m") for chaves in NIVEIS_MEDIANA]


def idade_relativa(d):
    tipica = pd.Series(np.nan, index=d.index)
    for chaves, m in zip(NIVEIS_MEDIANA, medianas):
        tipica = tipica.fillna(d[chaves].join(m, on=chaves)["m"])
    return d["idade_ingresso"] - tipica


for d in (treino, validacao, teste):
    d["idade_relativa"] = idade_relativa(d)
print(treino.groupby("nivel")["idade_relativa"].describe().round(1).to_string())

# %% [markdown]
# ### 1.2 Demanda do curso: diagnóstico antes de usar
#
# Inscritos e vagas são do ano-base da edição, não do ano de ingresso do
# aluno. Antes de usar, confere-se quanto falta e se o valor é estável dentro
# de um mesmo curso e campus.

# %%
if TEM_DEMANDA:
    for d in (treino, validacao, teste):
        d["log_inscritos_por_vaga"] = np.log1p(d["inscritos_por_vaga"])
    print(f"Sem relação inscritos/vaga (vagas = 0 ou ausente): "
          f"{treino['inscritos_por_vaga'].isna().mean():.1%} do treino")
    print(treino["inscritos_por_vaga"].describe().round(2).to_string())
    variacao = (treino.groupby(["ano_base", "unidade_de_ensino", "nome_de_curso", "turno"])
                      ["inscritos_por_vaga"].nunique())
    print(f"Grupos (edição, campus, curso, turno) com mais de um valor: {(variacao > 1).mean():.1%}")
    faixa = pd.qcut(treino["inscritos_por_vaga"], 5, duplicates="drop")
    print("\nEvasão por quintil de inscritos por vaga (treino):")
    print(treino.groupby(faixa, observed=True)["evadiu"].agg(["size", "mean"]).round(3).to_string())
    print(f"Evasão quando a relação está ausente: "
          f"{treino.loc[treino['inscritos_por_vaga'].isna(), 'evadiu'].mean():.3f}")

# %% [markdown]
# ### 1.3 Comparação na validação

# %%
def pipeline(nums):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), CAT),
        ("num", "passthrough", nums)])
    modelo = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CAT))), max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, l2_regularization=1.0,
        random_state=SEMENTE, **PARAMS)
    return Pipeline([("pre", pre), ("modelo", modelo)])


VARIANTES = {"atual": NUM, "+ idade relativa": NUM + ["idade_relativa"]}
if TEM_DEMANDA:
    VARIANTES["+ demanda"] = NUM + ["log_inscritos_por_vaga"]
    VARIANTES["+ idade relativa + demanda"] = NUM + ["idade_relativa", "log_inscritos_por_vaga"]

modelos, prob_va, linhas = {}, {}, []
for nome, nums in VARIANTES.items():
    modelos[nome] = pipeline(nums).fit(treino[CAT + nums], y_tr)
    prob_va[nome] = modelos[nome].predict_proba(validacao[CAT + nums])[:, 1]
    jovens = (validacao["idade_ingresso"] <= 17).values
    linhas.append({"variante": nome,
                   "AUC validação": roc_auc_score(y_va, prob_va[nome]),
                   "AUC até 17 anos": roc_auc_score(y_va[jovens], prob_va[nome][jovens]),
                   "Brier": brier_score_loss(y_va, prob_va[nome])})
comparacao = pd.DataFrame(linhas).set_index("variante")
comparacao["ganho AUC"] = comparacao["AUC validação"] - comparacao.loc["atual", "AUC validação"]
print(comparacao.round(4).to_string())


# %% [markdown]
# Escolha: a variante de maior AUC na validação, desde que o ganho sobre a
# atual seja de pelo menos 0,002 (abaixo disso é ruído, como mostrou a busca
# de hiperparâmetros). Se nenhuma passar, o modelo atual fica.

# %%
melhor = comparacao["AUC validação"].idxmax()
ESCOLHIDA = melhor if comparacao.loc[melhor, "ganho AUC"] >= 0.002 else "atual"
print(f"Variante escolhida: {ESCOLHIDA}")

# %% [markdown]
# ## 2. Limiar por nível de curso
#
# Hoje há um limiar único. Aqui se escolhe, na validação, um limiar para o
# técnico e outro para a graduação, cada um maximizando a acurácia
# balanceada dentro do seu nível. As probabilidades são calibradas antes
# (isotônica na validação), como no modelo principal.

# %%
calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prob_va[ESCOLHIDA], y_va)
pcal_va = calibrador.predict(prob_va[ESCOLHIDA])
GRADE = np.linspace(0.05, 0.95, 91)


def limiar_bal(y, p):
    return float(GRADE[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in GRADE]))])


LIMIAR_UNICO = limiar_bal(y_va, pcal_va)
LIMIARES_NIVEL = {nivel: limiar_bal(y_va[(validacao["nivel"] == nivel).values],
                                    pcal_va[(validacao["nivel"] == nivel).values])
                  for nivel in ["Técnico", "Graduação"]}
print(f"Limiar único: {LIMIAR_UNICO:.2f} · por nível: {LIMIARES_NIVEL}")


def decisoes(d, pcal, regra):
    lim = np.full(len(d), LIMIAR_UNICO) if regra == "único" else d["nivel"].map(LIMIARES_NIVEL).values
    return (pcal >= lim).astype(int)


def tabela_decisoes(d, y, pcal):
    linhas = []
    grupos = {"Todos": np.ones(len(d), bool),
              "Técnico": (d["nivel"] == "Técnico").values,
              "Graduação": (d["nivel"] == "Graduação").values,
              "Até 17 anos": (d["idade_ingresso"] <= 17).values}
    for regra in ["único", "por nível"]:
        alerta = decisoes(d, pcal, regra)
        for g, m in grupos.items():
            yy, aa = y[m], alerta[m]
            linhas.append({"regra": regra, "grupo": g, "n": int(m.sum()),
                           "taxa de alerta": aa.mean(),
                           "recall": aa[yy == 1].mean(),
                           "falsos positivos": aa[yy == 0].mean(),
                           "acurácia balanceada": balanced_accuracy_score(yy, aa)})
    return pd.DataFrame(linhas).set_index(["grupo", "regra"]).sort_index()


print(tabela_decisoes(validacao, y_va, pcal_va).round(3).to_string())

# %% [markdown]
# ## 3. Curva de aprendizado
#
# Treina a variante escolhida com frações do treino (amostras estratificadas)
# e mede a AUC na validação. Se a curva ainda sobe no fim, mais dados
# ajudariam; se achatou, o limite é a informação disponível.

# %%
FRACOES = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SEMENTES = [SEMENTE, SEMENTE + 1, SEMENTE + 2]
nums = VARIANTES[ESCOLHIDA]
curva = []
for f in FRACOES:
    for s in (SEMENTES if f < 1 else [SEMENTE]):
        if f < 1:
            sub, _ = train_test_split(treino, train_size=f, stratify=treino["evadiu"], random_state=s)
        else:
            sub = treino
        m = pipeline(nums).fit(sub[CAT + nums], sub["evadiu"])
        curva.append({"fração": f, "n": len(sub), "semente": s,
                      "AUC": roc_auc_score(y_va, m.predict_proba(validacao[CAT + nums])[:, 1])})
curva = pd.DataFrame(curva)
resumo_curva = curva.groupby(["fração", "n"])["AUC"].agg(["mean", "min", "max"]).reset_index()
print(resumo_curva.round(4).to_string(index=False))
curva.to_csv(PASTA_RES / "curva_aprendizado.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.fill_between(resumo_curva["n"], resumo_curva["min"], resumo_curva["max"], color=COR, alpha=0.15)
ax.plot(resumo_curva["n"], resumo_curva["mean"], "o-", color=COR)
for _, l in resumo_curva.iterrows():
    ax.annotate(f"{l['mean']:.3f}", (l["n"], l["mean"]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Matrículas de treino (escala log)")
ax.set_ylabel("AUC na validação")
ax.set_title("Curva de aprendizado do modelo principal")
fig.tight_layout()
fig.savefig(PASTA_FIG / "40_curva_aprendizado.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Resumo das escolhas
#
# Atributos: a variante escolhida na seção 1.3. Limiar: por nível se ele
# aumentar a acurácia balanceada média dos dois níveis na validação; senão,
# o único.

# %%
dec_va = tabela_decisoes(validacao, y_va, pcal_va)
bal_niveis = {regra: dec_va.loc[[("Técnico", regra), ("Graduação", regra)], "acurácia balanceada"].mean()
              for regra in ["único", "por nível"]}
REGRA = "por nível" if bal_niveis["por nível"] > bal_niveis["único"] + 0.002 else "único"
print(f"Acurácia balanceada média dos níveis: {bal_niveis} → regra escolhida: {REGRA}")

# %% [markdown]
# ## 5. Teste (uma única vez)
#
# Compara o modelo atual (limiar único) com a configuração escolhida. A
# diferença de AUC tem intervalo de confiança por bootstrap pareado: as
# mesmas reamostragens do teste para os dois modelos.

# %%
cal_atual = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prob_va["atual"], y_va)
p_atual = cal_atual.predict(modelos["atual"].predict_proba(teste[CAT + NUM])[:, 1])
p_nova = calibrador.predict(modelos[ESCOLHIDA].predict_proba(teste[CAT + VARIANTES[ESCOLHIDA]])[:, 1])

rng = np.random.default_rng(SEMENTE)
difs = []
for _ in range(1000):
    i = rng.integers(0, len(y_te), len(y_te))
    difs.append(roc_auc_score(y_te[i], p_nova[i]) - roc_auc_score(y_te[i], p_atual[i]))
dif_ic = np.percentile(difs, [2.5, 97.5])
print(f"AUC teste · atual: {roc_auc_score(y_te, p_atual):.4f} · {ESCOLHIDA}: {roc_auc_score(y_te, p_nova):.4f}")
print(f"Diferença: {roc_auc_score(y_te, p_nova) - roc_auc_score(y_te, p_atual):+.4f} "
      f"(IC 95%: {dif_ic[0]:+.4f} a {dif_ic[1]:+.4f})")
print(f"Brier · atual: {brier_score_loss(y_te, p_atual):.4f} · {ESCOLHIDA}: {brier_score_loss(y_te, p_nova):.4f}")

final = tabela_decisoes(teste, y_te, p_nova)
print(f"\nDecisões no teste ({ESCOLHIDA}):")
print(final.round(3).to_string())
final.to_csv(PASTA_RES / "experimentos_decisoes_teste.csv")
comparacao.to_csv(PASTA_RES / "experimentos_atributos_validacao.csv")

# %% [markdown]
# ## 6. Salvar a configuração final
#
# Só substitui o modelo da demonstração se algo mudou.

# %%
if ESCOLHIDA != "atual" or REGRA != "único":
    joblib.dump({"modelo": modelos[ESCOLHIDA], "calibrador": calibrador,
                 "limiar": LIMIAR_UNICO if REGRA == "único" else LIMIARES_NIVEL,
                 "regra_limiar": REGRA, "colunas": CAT + VARIANTES[ESCOLHIDA],
                 "variante": ESCOLHIDA, "hiperparametros": PARAMS,
                 "medianas_idade": medianas if "idade_relativa" in VARIANTES[ESCOLHIDA] else None},
                PASTA_RES / "modelo_final.joblib")
    print("Salvo resultados/modelo_final.joblib")
else:
    print("Nada mudou: a demonstração continua com resultados/modelo_principal.joblib")
