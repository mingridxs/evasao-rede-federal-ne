# %% [markdown]
# # Retidos: quem ainda estava em curso depois do prazo
#
# A base principal deixa de fora os alunos "Em curso" depois do fim previsto do
# ciclo (retidos), porque o desfecho deles ainda não existe. Esta análise de
# sensibilidade responde a três perguntas, sem mudar o modelo nem o site:
#   1. Quantos são os retidos e onde estão?
#   2. Que risco o modelo atual dá a eles? Se for parecido com o de quem
#      evadiu, eles se parecem mais com quem sai do que com quem conclui.
#   3. Um modelo para "não concluiu no prazo" (evadiu ou ficou retido) acerta
#      tanto quanto o atual? E o que se perde na pergunta original?
#
# Antes, gere as bases com retidos (uma vez por edição):
#   for ano in 2021 2022 2023; do
#     python scripts/01_preprocessa_pnp.py dados/brutos/microdados_eficiencia_academica_$ano.csv \
#         --ano $ano --regiao Nordeste --manter-retidos --saida dados/processados/ef_${ano}_com_retidos.csv
#   done
# Depois, na raiz do repositório:
#   python scripts/15_retidos.py

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from padroniza import padroniza_nomes

SEMENTE = 42
PASTA_DADOS, PASTA_RES, PASTA_FIG = Path("dados/processados"), Path("resultados"), Path("figuras")
PASTA_FIG.mkdir(exist_ok=True)
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

final = joblib.load(PASTA_RES / "modelo_final.joblib")
pre = final["modelo"].named_steps["pre"]
CAT, NUM = list(pre.transformers_[0][2]), list(pre.transformers_[1][2])
PARAMS, LIMIAR = final["hiperparametros"], final["limiar"]


def prepara(df):
    df = df.copy()
    numericas = {"carga_horaria", "ano_ingresso", "mes_ingresso", "duracao_prevista_meses", "idade_ingresso",
                 "ano_base", "evadiu", "retido", "inscritos", "vagas", "inscritos_por_vaga"}
    categoricas = [c for c in df.columns if c not in numericas]
    df[categoricas] = df[categoricas].astype(str)
    for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
        df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
    df["nivel"] = np.where(df["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
    return df


principal = prepara(pd.read_csv(PASTA_DADOS / "ef_2021_2023.csv"))
arquivos = [PASTA_DADOS / f"ef_{ano}_com_retidos.csv" for ano in (2021, 2022, 2023)]
faltam = [str(a) for a in arquivos if not a.exists()]
if faltam:
    raise SystemExit(f"Faltam as bases com retidos: {faltam}. Veja o comando no início deste script.")
com_retidos = prepara(padroniza_nomes(pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True), verbose=False))
retidos = com_retidos[com_retidos["retido"] == 1].reset_index(drop=True)
print(f"Base principal: {len(principal):,} matrículas com desfecho · "
      f"nas bases com retidos, {int((com_retidos['retido'] == 0).sum()):,} com desfecho (devem ser iguais)")

# %% [markdown]
# ## 1. Quantos são e onde estão

# %%
total = len(principal) + len(retidos)
print(f"Retidos: {len(retidos):,} ({len(retidos) / total:.1%} das {total:,} matrículas com ciclo encerrado)\n")
juntos = pd.concat([principal.assign(situacao=np.where(principal["evadiu"] == 1, "Evadiu", "Concluiu")),
                    retidos.assign(situacao="Retido")], ignore_index=True)
for coluna, titulo in [("ano_base", "edição"), ("nivel", "nível"), ("tipo_de_curso", "tipo de curso"),
                       ("tipo_de_oferta", "forma de oferta")]:
    tab = pd.crosstab(juntos[coluna], juntos["situacao"], normalize="index").mul(100).round(1)
    tab["n"] = juntos[coluna].value_counts()
    print(f"Situação por {titulo} (%):\n{tab.sort_values('n', ascending=False).to_string()}\n")
perfil = juntos.groupby("situacao").agg(idade_mediana=("idade_ingresso", "median"),
                                         sem_renda=("renda_nao_declarada", "mean"),
                                         sem_cor=("cor_nao_declarada", "mean"))
perfil[["sem_renda", "sem_cor"]] *= 100
print("Perfil por situação (idade mediana; % sem renda e sem cor/raça declaradas):")
print(perfil.round(1).to_string())

# %% [markdown]
# ## 2. Que risco o modelo atual dá aos retidos
#
# Os retidos nunca foram vistos pelo modelo. Compara-se com quem evadiu e quem
# concluiu na partição de teste, que o modelo também não viu.

# %%
def risco(modelo, calibrador, cats, d):
    return calibrador.predict(modelo.predict_proba(d[cats + NUM])[:, 1])


teste = principal[principal["particao"] == "teste"]
grupos = {"Concluiu (teste)": teste[teste["evadiu"] == 0], "Evadiu (teste)": teste[teste["evadiu"] == 1],
          "Retido": retidos}
linhas, probs = [], {}
for nome, d in grupos.items():
    p = risco(final["modelo"], final["calibrador"], CAT, d)
    probs[nome] = p
    lim = d["nivel"].map(LIMIAR).to_numpy() if isinstance(LIMIAR, dict) else LIMIAR
    for nivel in ["Todos", "Técnico", "Graduação"]:
        m = np.ones(len(d), bool) if nivel == "Todos" else (d["nivel"] == nivel).to_numpy()
        if m.sum() == 0:
            continue
        linhas.append({"grupo": nome, "nível": nivel, "n": int(m.sum()), "risco médio (%)": p[m].mean() * 100,
                       "risco mediano (%)": np.median(p[m]) * 100,
                       "recebem alerta (%)": (p[m] >= np.broadcast_to(lim, p.shape)[m]).mean() * 100})
tab_risco = pd.DataFrame(linhas).set_index(["nível", "grupo"]).sort_index()
print(tab_risco.round(1).to_string())

fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.linspace(0, 1, 21)
for (nome, p), cor in zip(probs.items(), ["#1f4f8f", "#e08a2e", "#6b7a90"]):
    ax.hist(p, bins=bins, density=True, histtype="step", lw=2, color=cor, label=f"{nome} (n={len(p):,})".replace(",", "."))
ax.set_xlabel("Risco de evasão previsto pelo modelo atual")
ax.set_ylabel("Densidade")
ax.set_title("Risco previsto: retidos comparados a quem evadiu e a quem concluiu")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(PASTA_FIG / "42_retidos_risco.png", dpi=150)
plt.close(fig)
print("\nFigura: figuras/42_retidos_risco.png")

# %% [markdown]
# ## 3. Modelo para "não concluiu no prazo"
#
# Os retidos são divididos 70/20/10 do mesmo jeito que a base principal
# (estratificado por edição, semente 42) e somados a ela. O rótulo novo é
# 1 para quem evadiu ou ficou retido. O modelo usa os mesmos atributos e
# hiperparâmetros do modelo final.

# %%
idx_tr, idx_resto = train_test_split(retidos.index, test_size=0.30, stratify=retidos["ano_base"], random_state=SEMENTE)
idx_va, idx_te = train_test_split(idx_resto, test_size=1 / 3, stratify=retidos.loc[idx_resto, "ano_base"],
                                  random_state=SEMENTE)
retidos["particao"] = "treino"
retidos.loc[idx_va, "particao"] = "validacao"
retidos.loc[idx_te, "particao"] = "teste"
base2 = pd.concat([principal, retidos], ignore_index=True)
base2["nao_concluiu"] = ((base2["evadiu"] == 1) | (base2["retido"].fillna(0) == 1)).astype(int)
partes2 = {p: base2[base2["particao"] == p] for p in ["treino", "validacao", "teste"]}
print("Taxa de 'não concluiu no prazo':", {p: f"{d['nao_concluiu'].mean():.1%}" for p, d in partes2.items()})

pre2 = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), CAT),
    ("num", "passthrough", NUM)])
modelo2 = Pipeline([("pre", pre2), ("modelo", HistGradientBoostingClassifier(
    categorical_features=list(range(len(CAT))), max_iter=1000, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, l2_regularization=1.0, random_state=SEMENTE, **PARAMS))])
modelo2.fit(partes2["treino"][CAT + NUM], partes2["treino"]["nao_concluiu"])
cal2 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
    modelo2.predict_proba(partes2["validacao"][CAT + NUM])[:, 1], partes2["validacao"]["nao_concluiu"])
print(f"Modelo 'não concluiu no prazo': {modelo2.named_steps['modelo'].n_iter_} árvores")

rng = np.random.default_rng(SEMENTE)


def auc_ic(y, p, n=500):
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(v, 2.5), np.percentile(v, 97.5)


te2 = partes2["teste"]
comparacao = []
for nome_modelo, (m, c) in {"atual (evasão)": (final["modelo"], final["calibrador"]),
                             "não concluiu no prazo": (modelo2, cal2)}.items():
    for pergunta, dados, y in [("não concluiu no prazo (teste com retidos)", te2, te2["nao_concluiu"]),
                               ("evasão (teste original)", teste, teste["evadiu"])]:
        auc, inf, sup = auc_ic(y, risco(m, c, CAT, dados))
        comparacao.append({"modelo": nome_modelo, "pergunta avaliada": pergunta, "n": len(dados),
                           "AUC": auc, "IC 95%": f"{inf:.3f}–{sup:.3f}"})
comparacao = pd.DataFrame(comparacao).set_index(["pergunta avaliada", "modelo"]).sort_index()
print(comparacao.round(4).to_string())

tab_risco.round(3).to_csv(PASTA_RES / "retidos_risco.csv")
comparacao.round(4).to_csv(PASTA_RES / "retidos_modelo.csv")
print("\nSalvos resultados/retidos_risco.csv e resultados/retidos_modelo.csv")
