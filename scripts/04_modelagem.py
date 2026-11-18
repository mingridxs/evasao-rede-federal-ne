# %% [markdown]
# # Modelagem — baselines, modelo principal e avaliação
#
# Protocolo:
# - **Treino** (70%): ajuste de todos os modelos e codificadores.
# - **Validação** (20%): escolha de hiperparâmetros, calibração das
#   probabilidades e escolha do limiar de decisão.
# - **Teste** (10%): usado uma única vez, na seção 6, depois de tudo decidido.
#
# Três conjuntos de atributos (decisões de 17/11 e 18/11):
# - **sem indicadores:** dados do curso, da oferta, sexo e idade no ingresso
#   (era o modelo principal até 18/11);
# - **principal:** o anterior mais dois indicadores, "não declarou renda" e
#   "não declarou cor/raça". O teste de 18/11 (scripts/10_renda_raca.py)
#   mostrou que esses registros são feitos no ingresso e que todo o ganho de
#   renda e cor/raça vem deles;
# - **completo:** sem indicadores mais os valores de renda e cor/raça, com
#   "não declarado" como categoria. Fica como comparação.
#
# Saídas: `resultados/metricas_validacao.csv`, `resultados/metricas_teste.csv`,
# `resultados/predicoes_teste.csv` (para a análise de viés),
# `resultados/modelo_principal.joblib` (para a demonstração) e figuras em `figuras/`.

# %%
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEMENTE = 42
np.random.seed(SEMENTE)
PASTA_FIG, PASTA_RES = Path("figuras"), Path("resultados")
PASTA_FIG.mkdir(exist_ok=True)
PASTA_RES.mkdir(exist_ok=True)
print(f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}")

COR, CINZA = "#2b6cb0", "#9aa5b1"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

# %% [markdown]
# ## 1. Dados e atributos
#
# Ficam fora: `ano_base` (é a edição do arquivo, não uma característica do
# aluno), `ano_ingresso` (na edição, é determinado pelo fim do ciclo e pela
# duração, e não generaliza para turmas futuras), `regiao` (constante) e
# `particao`.

# %%
CAT_PRINCIPAL = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso",
                 "tipo_de_oferta", "modalidade_de_ensino", "eixo_tecnologico",
                 "subeixo_tecnologico", "turno", "fonte_de_financiamento", "sexo"]
NUM = ["carga_horaria", "duracao_prevista_meses", "mes_ingresso", "idade_ingresso"]
CAT_EXTRA = ["cor_raca", "renda_familiar"]
IND = ["renda_nao_declarada", "cor_nao_declarada"]
CONJUNTOS = {"sem indicadores": (CAT_PRINCIPAL, NUM),
             "principal": (CAT_PRINCIPAL, NUM + IND),
             "completo": (CAT_PRINCIPAL + CAT_EXTRA, NUM)}
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}

df = pd.read_csv("dados/processados/ef_2021_2023.csv",
                 dtype={c: str for c in CAT_PRINCIPAL + CAT_EXTRA})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
treino = df[df["particao"] == "treino"].reset_index(drop=True)
validacao = df[df["particao"] == "validacao"].reset_index(drop=True)
teste = df[df["particao"] == "teste"].reset_index(drop=True)
y_tr, y_va, y_te = treino["evadiu"].values, validacao["evadiu"].values, teste["evadiu"].values
print({nome: len(p) for nome, p in [("treino", treino), ("validacao", validacao), ("teste", teste)]})


# %% [markdown]
# ## 2. Funções de avaliação

# %%
def metricas(y, prob, limiar=0.5):
    """Métricas de ranqueamento, de probabilidade e de decisão (com o limiar dado)."""
    pred = (prob >= limiar).astype(int)
    return {
        "AUC": roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5,
        "AP": average_precision_score(y, prob),
        "Brier": brier_score_loss(y, prob),
        "LogLoss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)),
        "F1": f1_score(y, pred, zero_division=0),
        "F1 macro": f1_score(y, pred, average="macro", zero_division=0),
        "Precisão": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Acurácia": accuracy_score(y, pred),
        "Acurácia balanceada": balanced_accuracy_score(y, pred),
        "Limiar": limiar,
    }


def melhor_limiar(y, prob):
    """Limiar que maximiza a acurácia balanceada (escolhido na validação).

    Não se usa o F1 da classe 'evadiu' porque, com 58% de evasão, ele é
    maximizado prevendo evasão para todos, o que não serve para nada."""
    grade = np.linspace(0.05, 0.95, 91)
    bal = [balanced_accuracy_score(y, (prob >= t).astype(int)) for t in grade]
    return float(grade[int(np.argmax(bal))])


def auc_bootstrap(y, prob, n=500, semente=SEMENTE):
    """IC 95% da AUC por bootstrap."""
    rng = np.random.default_rng(semente)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], prob[idx]))
    return np.percentile(aucs, [2.5, 97.5])


# %% [markdown]
# ## 3. Baselines
#
# 1. **Classe majoritária:** prevê para todos a taxa de evasão do treino.
# 2. **Taxa histórica do curso:** a evasão do mesmo curso no mesmo campus,
#    no treino, suavizada em direção à taxa do tipo de curso (cursos pequenos
#    não ficam com taxas extremas). Não usa nenhum dado do aluno; é o que uma
#    coordenação já saberia sem modelo.
# 3. **Regressão logística** com one-hot nos categóricos.

# %%
class TaxaHistoricaCurso:
    """Taxa de evasão por (campus, curso) no treino, suavizada pela taxa do tipo de curso."""

    def __init__(self, m=20):
        self.m = m

    def fit(self, X, y):
        d = X[["unidade_de_ensino", "nome_de_curso", "tipo_de_curso"]].assign(y=y)
        self.geral_ = d["y"].mean()
        self.tipo_ = d.groupby("tipo_de_curso")["y"].mean()
        g = d.groupby(["unidade_de_ensino", "nome_de_curso"]).agg(
            n=("y", "size"), soma=("y", "sum"), tipo=("tipo_de_curso", "first"))
        prior = g["tipo"].map(self.tipo_).fillna(self.geral_)
        self.curso_ = (g["soma"] + self.m * prior) / (g["n"] + self.m)
        return self

    def predict_proba(self, X):
        chave = pd.MultiIndex.from_frame(X[["unidade_de_ensino", "nome_de_curso"]])
        p = pd.Series(self.curso_.reindex(chave).values, index=X.index)
        p = p.fillna(X["tipo_de_curso"].map(self.tipo_)).fillna(self.geral_)
        return np.column_stack([1 - p.values, p.values])


def pipeline_logistica(cats, nums):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cats),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), nums),
    ])
    return Pipeline([("pre", pre),
                     ("modelo", LogisticRegression(max_iter=3000, C=1.0))])


def pipeline_boosting(cats, nums, **params):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cats),
        ("num", "passthrough", nums),
    ])
    modelo = HistGradientBoostingClassifier(
        categorical_features=list(range(len(cats))), max_iter=1000,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=SEMENTE, **params)
    return Pipeline([("pre", pre), ("modelo", modelo)])


# %%
modelos = {}           # nome -> objeto treinado
prob_va = {}           # nome -> probabilidades na validação

modelos["Classe majoritária"] = None
prob_va["Classe majoritária"] = np.full(len(validacao), y_tr.mean())

modelos["Taxa histórica do curso"] = TaxaHistoricaCurso().fit(treino, y_tr)
prob_va["Taxa histórica do curso"] = modelos["Taxa histórica do curso"].predict_proba(validacao)[:, 1]

for conjunto, (cats, nums) in CONJUNTOS.items():
    nome = f"Regressão logística ({conjunto})"
    inicio = time.time()
    modelos[nome] = pipeline_logistica(cats, nums).fit(treino[cats + nums], y_tr)
    prob_va[nome] = modelos[nome].predict_proba(validacao[cats + nums])[:, 1]
    print(f"{nome}: {time.time() - inicio:.0f}s, AUC validação {roc_auc_score(y_va, prob_va[nome]):.4f}")

# %% [markdown]
# ## 4. Modelo principal: gradient boosting
#
# Busca em grade pequena, escolhida pela AUC na validação. O próprio modelo
# separa 10% do treino para parar cedo (early stopping), então a validação
# externa não participa do ajuste dos pesos.

# %%
GRADE = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
         "min_samples_leaf": [20, 100]}
combinacoes = [dict(zip(GRADE, v)) for v in product(*GRADE.values())]

busca = []
for conjunto, (cats, nums) in CONJUNTOS.items():
    for params in combinacoes:
        inicio = time.time()
        pipe = pipeline_boosting(cats, nums, l2_regularization=1.0, **params)
        pipe.fit(treino[cats + nums], y_tr)
        p = pipe.predict_proba(validacao[cats + nums])[:, 1]
        busca.append({"conjunto": conjunto, **params,
                      "iteracoes": pipe.named_steps["modelo"].n_iter_,
                      "AUC validação": roc_auc_score(y_va, p),
                      "segundos": round(time.time() - inicio, 1)})
        print(busca[-1])

busca = pd.DataFrame(busca).sort_values(["conjunto", "AUC validação"], ascending=[True, False])
busca.to_csv(PASTA_RES / "busca_hiperparametros.csv", index=False)
print(busca.groupby("conjunto").head(3).to_string(index=False))

# %%
melhores = {}
for conjunto, (cats, nums) in CONJUNTOS.items():
    linha = busca[busca["conjunto"] == conjunto].iloc[0]
    params = {"learning_rate": float(linha["learning_rate"]),
              "max_leaf_nodes": int(linha["max_leaf_nodes"]),
              "min_samples_leaf": int(linha["min_samples_leaf"])}
    melhores[conjunto] = params
    nome = f"Gradient boosting ({conjunto})"
    modelos[nome] = pipeline_boosting(cats, nums, l2_regularization=1.0, **params).fit(
        treino[cats + nums], y_tr)
    prob_va[nome] = modelos[nome].predict_proba(validacao[cats + nums])[:, 1]
    print(f"{nome}: {params}, AUC validação {roc_auc_score(y_va, prob_va[nome]):.4f}")

# %% [markdown]
# ## 5. Calibração e limiar (na validação)
#
# A saída do sistema é uma probabilidade, então ela precisa ser confiável:
# se o modelo diz 70%, cerca de 70% daqueles alunos devem evadir. A
# calibração isotônica é ajustada nas previsões da validação; por ser
# monótona, não altera a AUC. O limiar de decisão é o que maximiza a acurácia
# balanceada (média do recall das duas classes) na validação, e é aplicado
# sem mudança no teste.

# %%
def colunas_de(nome):
    if nome in ("Classe majoritária", "Taxa histórica do curso"):
        return None
    for conjunto, (cats, nums) in CONJUNTOS.items():
        if f"({conjunto})" in nome:
            return cats + nums
    raise KeyError(nome)


calibradores, limiares = {}, {}
for nome, p in list(prob_va.items()):
    if nome.startswith("Gradient boosting") or nome.startswith("Regressão"):
        calibradores[nome] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y_va)
        prob_va[nome + " · calibrado"] = calibradores[nome].predict(p)
for nome, p in prob_va.items():
    limiares[nome] = melhor_limiar(y_va, p)

tabela_va = pd.DataFrame({nome: metricas(y_va, p, limiares[nome]) for nome, p in prob_va.items()}).T
tabela_va.to_csv(PASTA_RES / "metricas_validacao.csv")
print(tabela_va.round(4).sort_values("AUC").to_string())

# %%
fig, ax = plt.subplots(figsize=(5.5, 5))
for nome, cor, estilo in [("Gradient boosting (principal)", CINZA, "--"),
                          ("Gradient boosting (principal) · calibrado", COR, "-")]:
    frac, media = calibration_curve(y_va, prob_va[nome], n_bins=10, strategy="quantile")
    ax.plot(media, frac, estilo, marker="o", color=cor,
            label=nome.replace("Gradient boosting (principal)", "Antes").replace("Antes · calibrado", "Depois da calibração"))
ax.plot([0, 1], [0, 1], ":", color="#555", label="Calibração perfeita")
ax.set_xlabel("Probabilidade prevista de evasão")
ax.set_ylabel("Proporção que evadiu")
ax.set_title("Calibração do modelo principal (validação)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PASTA_FIG / "20_calibracao_validacao.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Importância dos atributos (validação)
#
# Importância por permutação: quanto a AUC cai quando os valores de um
# atributo são embaralhados. Calculada numa amostra de 20 mil matrículas da
# validação para economizar tempo.

# %%
amostra = validacao.sample(min(20000, len(validacao)), random_state=SEMENTE)
cols = CONJUNTOS["principal"][0] + CONJUNTOS["principal"][1]
imp = permutation_importance(modelos["Gradient boosting (principal)"], amostra[cols],
                             amostra["evadiu"], scoring="roc_auc", n_repeats=5,
                             random_state=SEMENTE, n_jobs=-1)
importancia = pd.DataFrame({"atributo": cols, "queda_auc": imp.importances_mean,
                            "dp": imp.importances_std}).sort_values("queda_auc")
importancia.to_csv(PASTA_RES / "importancia_permutacao.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 0.35 * len(importancia) + 1.2))
ax.barh(importancia["atributo"], importancia["queda_auc"], xerr=importancia["dp"],
        color=[COR if v == importancia["queda_auc"].max() else CINZA for v in importancia["queda_auc"]])
ax.set_xlabel("Queda da AUC ao embaralhar o atributo")
ax.set_title("O que o modelo principal usa")
fig.tight_layout()
fig.savefig(PASTA_FIG / "21_importancia.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Avaliação final no teste
#
# Executada uma vez, com tudo o que foi decidido acima congelado: modelos,
# hiperparâmetros, calibradores e limiares.

# %%
prob_te = {}
prob_te["Classe majoritária"] = np.full(len(teste), y_tr.mean())
prob_te["Taxa histórica do curso"] = modelos["Taxa histórica do curso"].predict_proba(teste)[:, 1]
for nome, modelo in modelos.items():
    cols = colunas_de(nome)
    if cols is None:
        continue
    prob_te[nome] = modelo.predict_proba(teste[cols])[:, 1]
    if nome in calibradores:
        prob_te[nome + " · calibrado"] = calibradores[nome].predict(prob_te[nome])

linhas = {}
for nome, p in prob_te.items():
    m = metricas(y_te, p, limiares[nome])
    constante = len(np.unique(p)) == 1
    m["AUC IC inf"], m["AUC IC sup"] = (m["AUC"], m["AUC"]) if constante else auc_bootstrap(y_te, p)
    linhas[nome] = m
tabela_te = pd.DataFrame(linhas).T
tabela_te.to_csv(PASTA_RES / "metricas_teste.csv")
print(tabela_te.round(4).sort_values("AUC").to_string())

# %%
ordem = ["Classe majoritária", "Taxa histórica do curso", "Regressão logística (sem indicadores)",
         "Gradient boosting (sem indicadores) · calibrado", "Regressão logística (principal)",
         "Gradient boosting (principal) · calibrado", "Gradient boosting (completo) · calibrado"]
t = tabela_te.loc[ordem]
fig, ax = plt.subplots(figsize=(8, 4.4))
cores = [COR if n == "Gradient boosting (principal) · calibrado" else CINZA for n in ordem]
ax.barh(range(len(t)), t["AUC"], color=cores,
        xerr=[np.clip(t["AUC"] - t["AUC IC inf"], 0, None), np.clip(t["AUC IC sup"] - t["AUC"], 0, None)],
        error_kw={"ecolor": "#555", "capsize": 3})
ax.set_yticks(range(len(t)), [n.replace(" · calibrado", "") for n in ordem])
ax.invert_yaxis()
for y, (auc, sup) in enumerate(zip(t["AUC"], t["AUC IC sup"])):
    ax.text(sup + 0.005, y, f"{auc:.3f}", va="center", fontsize=8)
ax.axvline(0.5, ls=":", color="#555")
ax.set_xlim(0.45, 1.0)
ax.set_xlabel("AUC no teste · barras: IC 95% por bootstrap · pontilhado: acaso")
ax.set_title("Comparação dos modelos no teste")
fig.tight_layout()
fig.savefig(PASTA_FIG / "22_comparacao_auc_teste.png", bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5.5, 5))
for nome, cor in [("Taxa histórica do curso", CINZA), ("Regressão logística (principal)", "#7f9cc7"),
                  ("Gradient boosting (principal) · calibrado", COR)]:
    fpr, tpr, _ = roc_curve(y_te, prob_te[nome])
    ax.plot(fpr, tpr, color=cor, label=f"{nome.replace(' · calibrado', '')} ({tabela_te.loc[nome, 'AUC']:.3f})")
ax.plot([0, 1], [0, 1], ":", color="#555")
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
ax.set_title("Curvas ROC no teste")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(PASTA_FIG / "23_roc_teste.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Salvar para as próximas etapas

# %%
saida = teste[["instituicao", "uf", "tipo_de_curso", "sexo", "cor_raca", "renda_familiar",
               "idade_ingresso", "ano_base", "evadiu"]].copy()
saida["prob_principal"] = prob_te["Gradient boosting (principal) · calibrado"]
saida["prob_completo"] = prob_te["Gradient boosting (completo) · calibrado"]
saida["prob_sem_indicadores"] = prob_te["Gradient boosting (sem indicadores) · calibrado"]
saida["prob_taxa_curso"] = prob_te["Taxa histórica do curso"]
saida["limiar_principal"] = limiares["Gradient boosting (principal) · calibrado"]
saida.to_csv(PASTA_RES / "predicoes_teste.csv", index=False)

joblib.dump({"modelo": modelos["Gradient boosting (principal)"],
             "calibrador": calibradores["Gradient boosting (principal)"],
             "limiar": limiares["Gradient boosting (principal) · calibrado"],
             "colunas": CONJUNTOS["principal"][0] + CONJUNTOS["principal"][1],
             "hiperparametros": melhores["principal"]},
            PASTA_RES / "modelo_principal.joblib")
print("Salvos: predicoes_teste.csv e modelo_principal.joblib em resultados/")
