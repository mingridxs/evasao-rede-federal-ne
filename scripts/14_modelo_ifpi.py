# %% [markdown]
# # Modelo do Nordeste ou modelo só do IFPI?
#
# Pergunta: para usar dentro do Instituto Federal do Piauí, é melhor o modelo
# treinado com todo o Nordeste (que já recebe instituição e campus como
# entrada) ou um modelo treinado só com matrículas do IFPI?
#
# Modelos comparados, todos com os hiperparâmetros do modelo final:
#   A. Nordeste: o modelo final, como está no site.
#   B. Só IFPI: treinado no treino do IFPI e calibrado na validação do IFPI.
#   C. Nordeste recalibrado no IFPI: as árvores de A, com a calibração
#      refeita na validação do IFPI (corrige um eventual desvio de nível).
#   D. Nordeste sem instituição e campus: mostra quanto o modelo ganha por
#      saber a instituição.
#
# Todos são avaliados só nas matrículas do IFPI do teste. Os limiares de
# alerta (um por nível) são escolhidos na validação do IFPI, para todos os
# modelos, maximizando a acurácia balanceada. O teste do IFPI é pequeno
# (cerca de 2 mil matrículas), então as diferenças vêm com intervalo de
# confiança por bootstrap pareado.
#
# Uso, na raiz do repositório, depois do script 06:
#     python scripts/14_modelo_ifpi.py

# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEMENTE = 42
IFPI = "Instituto Federal do Piauí"
PASTA_RES = Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

final = joblib.load(PASTA_RES / "modelo_final.joblib")
pre = final["modelo"].named_steps["pre"]
CAT, NUM = list(pre.transformers_[0][2]), list(pre.transformers_[1][2])
PARAMS = final["hiperparametros"]
print("Atributos:", CAT + NUM)
print("Hiperparâmetros:", PARAMS)

df = pd.read_csv("dados/processados/ef_2021_2023.csv", dtype={c: str for c in CAT + ["renda_familiar"]})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
df["nivel"] = np.where(df["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
partes = {p: df[df["particao"] == p].reset_index(drop=True) for p in ["treino", "validacao", "teste"]}
ifpi = {p: d[d["instituicao"] == IFPI].reset_index(drop=True) for p, d in partes.items()}
print({p: len(d) for p, d in ifpi.items()}, "matrículas do IFPI por partição")
if len(ifpi["teste"]) < 300:
    raise SystemExit(f"Poucas matrículas de '{IFPI}' no teste. Rode antes o script 02 com a padronização de nomes.")


def pipeline(cats):
    etapa = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cats),
        ("num", "passthrough", NUM)])
    return Pipeline([("pre", etapa), ("modelo", HistGradientBoostingClassifier(
        categorical_features=list(range(len(cats))), max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, l2_regularization=1.0,
        random_state=SEMENTE, **PARAMS))])


def calibrar(bruto, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bruto, y)


# %% [markdown]
# ## Treino dos modelos

# %%
bruto = lambda m, cats, d: m.predict_proba(d[cats + NUM])[:, 1]
prob = {}  # (modelo, partição do IFPI) -> probabilidade calibrada

# A. Nordeste, como no site
mA = final["modelo"]
for p in ["validacao", "teste"]:
    prob[("A. Nordeste", p)] = final["calibrador"].predict(bruto(mA, CAT, ifpi[p]))

# B. Só IFPI
mB = pipeline(CAT).fit(ifpi["treino"][CAT + NUM], ifpi["treino"]["evadiu"])
calB = calibrar(bruto(mB, CAT, ifpi["validacao"]), ifpi["validacao"]["evadiu"])
for p in ["validacao", "teste"]:
    prob[("B. Só IFPI", p)] = calB.predict(bruto(mB, CAT, ifpi[p]))
print(f"B. Só IFPI: {mB.named_steps['modelo'].n_iter_} árvores, treinado com {len(ifpi['treino']):,} matrículas")

# C. Nordeste recalibrado no IFPI
calC = calibrar(bruto(mA, CAT, ifpi["validacao"]), ifpi["validacao"]["evadiu"])
for p in ["validacao", "teste"]:
    prob[("C. Nordeste recalibrado no IFPI", p)] = calC.predict(bruto(mA, CAT, ifpi[p]))

# D. Nordeste sem instituição e campus
CAT_D = [c for c in CAT if c not in ("instituicao", "unidade_de_ensino")]
mD = pipeline(CAT_D).fit(partes["treino"][CAT_D + NUM], partes["treino"]["evadiu"])
calD = calibrar(bruto(mD, CAT_D, partes["validacao"]), partes["validacao"]["evadiu"])
for p in ["validacao", "teste"]:
    prob[("D. Nordeste sem instituição e campus", p)] = calD.predict(bruto(mD, CAT_D, ifpi[p]))

MODELOS = sorted({m for m, _ in prob})

# %% [markdown]
# ## Avaliação no teste do IFPI

# %%
rng = np.random.default_rng(SEMENTE)
te, va = ifpi["teste"], ifpi["validacao"]
y = te["evadiu"].to_numpy()
amostras = [rng.integers(0, len(te), len(te)) for _ in range(1000)]


def limiares(nome):
    """Um limiar por nível, escolhido na validação do IFPI."""
    grade = np.linspace(0.05, 0.95, 91)
    saida = {}
    for nivel in ["Técnico", "Graduação"]:
        m = (va["nivel"] == nivel).to_numpy()
        p, yv = prob[(nome, "validacao")][m], va["evadiu"].to_numpy()[m]
        saida[nivel] = float(grade[int(np.argmax([balanced_accuracy_score(yv, (p >= t).astype(int)) for t in grade]))])
    return saida


linhas = []
for nome in MODELOS:
    p = prob[(nome, "teste")]
    lim = limiares(nome)
    alerta = (p >= te["nivel"].map(lim).to_numpy()).astype(int)
    aucs = [roc_auc_score(y[i], p[i]) for i in amostras]
    linha = {"modelo": nome, "AUC": roc_auc_score(y, p),
             "AUC IC": f"{np.percentile(aucs, 2.5):.3f}–{np.percentile(aucs, 97.5):.3f}",
             "Brier": brier_score_loss(y, p),
             "erro de calibração (pontos)": (p.mean() - y.mean()) * 100,
             "acurácia": (alerta == y).mean(), "acurácia balanceada": balanced_accuracy_score(y, alerta),
             "limiar técnico": lim["Técnico"], "limiar graduação": lim["Graduação"]}
    for nivel in ["Técnico", "Graduação"]:
        m = (te["nivel"] == nivel).to_numpy()
        linha[f"erro calib. {nivel.lower()} (pontos)"] = (p[m].mean() - y[m].mean()) * 100
    linhas.append(linha)
tabela = pd.DataFrame(linhas).set_index("modelo")
print(f"Teste do IFPI: {len(te):,} matrículas, evasão real {y.mean():.1%}\n")
print(tabela.round(3).to_string())

print("\nDiferença de AUC em relação ao A (bootstrap pareado, IC 95%):")
pA = prob[("A. Nordeste", "teste")]
for nome in MODELOS:
    if nome == "A. Nordeste":
        continue
    p = prob[(nome, "teste")]
    dif = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], pA[i]) for i in amostras]
    print(f"  {nome}: {roc_auc_score(y, p) - roc_auc_score(y, pA):+.4f} "
          f"({np.percentile(dif, 2.5):+.4f} a {np.percentile(dif, 97.5):+.4f})")

tabela.round(4).to_csv(PASTA_RES / "modelo_ifpi.csv")
print("\nSalvo resultados/modelo_ifpi.csv")
