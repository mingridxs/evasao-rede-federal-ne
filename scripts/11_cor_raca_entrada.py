# %% [markdown]
# # Cor/raça como entrada do modelo: corrige o erro com estudantes indígenas?
#
# O modelo principal subestima em cerca de 10 pontos o risco de estudantes
# indígenas, porque não vê a cor/raça. Aqui se treina o mesmo modelo com a
# cor/raça como entrada e se compara, grupo por grupo:
# - calibração (probabilidade média prevista contra evasão real);
# - AUC dentro do grupo;
# - AUC geral.
#
# A decisão é tomada na validação. O teste aparece no fim só para relatar,
# e o intervalo de confiança da evasão real mostra quanto os grupos pequenos
# deixam de margem.
#
# Uso, na raiz do repositório: python scripts/11_cor_raca_entrada.py

# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEMENTE = 42
PASTA_RES = Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
CAT = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso", "tipo_de_oferta",
       "modalidade_de_ensino", "eixo_tecnologico", "subeixo_tecnologico", "turno",
       "fonte_de_financiamento", "sexo"]
NUM = ["carga_horaria", "duracao_prevista_meses", "mes_ingresso", "idade_ingresso",
       "renda_nao_declarada", "cor_nao_declarada"]

df = pd.read_csv("dados/processados/ef_2021_2023.csv", dtype={c: str for c in CAT + ["cor_raca", "renda_familiar"]})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
partes = {p: df[df["particao"] == p].reset_index(drop=True) for p in ["treino", "validacao", "teste"]}
PARAMS = joblib.load(PASTA_RES / "modelo_principal.joblib")["hiperparametros"]
print("Hiperparâmetros do modelo principal:", PARAMS)


def pipeline(cats):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cats),
        ("num", "passthrough", NUM)])
    return Pipeline([("pre", pre), ("modelo", HistGradientBoostingClassifier(
        categorical_features=list(range(len(cats))), max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, l2_regularization=1.0,
        random_state=SEMENTE, **PARAMS))])


# %% [markdown]
# ## Treino e calibração

# %%
VARIANTES = {"principal": CAT, "principal + cor/raça": CAT + ["cor_raca"]}
prob = {}
for nome, cats in VARIANTES.items():
    modelo = pipeline(cats).fit(partes["treino"][cats + NUM], partes["treino"]["evadiu"])
    bruto_va = modelo.predict_proba(partes["validacao"][cats + NUM])[:, 1]
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bruto_va, partes["validacao"]["evadiu"])
    for p in ["validacao", "teste"]:
        prob[(nome, p)] = calib.predict(modelo.predict_proba(partes[p][cats + NUM])[:, 1])


def wilson(k, n, z=1.96):
    q = k / n
    centro = (q + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(q * (1 - q) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - meia, centro + meia


def por_grupo(parte):
    d = partes[parte]
    linhas = []
    for grupo, idx in d.groupby("cor_raca").groups.items():
        g = d.loc[idx]
        inf, sup = wilson(g["evadiu"].sum(), len(g))
        linha = {"grupo": grupo, "n": len(g), "evasão real (%)": g["evadiu"].mean() * 100,
                 "IC 95% real": f"{inf * 100:.0f}–{sup * 100:.0f}"}
        for nome in VARIANTES:
            p = prob[(nome, parte)][idx]
            linha[f"previsto · {nome} (%)"] = p.mean() * 100
            linha[f"erro · {nome} (pontos)"] = (p.mean() - g["evadiu"].mean()) * 100
            if g["evadiu"].nunique() == 2 and g["evadiu"].sum() >= 30 and (1 - g["evadiu"]).sum() >= 30:
                linha[f"AUC · {nome}"] = roc_auc_score(g["evadiu"], p)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("grupo").sort_values("n", ascending=False)


# %% [markdown]
# ## Validação: base da decisão

# %%
geral = pd.DataFrame({
    nome: {"AUC": roc_auc_score(partes["validacao"]["evadiu"], prob[(nome, "validacao")]),
           "Brier": brier_score_loss(partes["validacao"]["evadiu"], prob[(nome, "validacao")])}
    for nome in VARIANTES}).T
print(geral.round(4).to_string())
va = por_grupo("validacao")
print("\nPor cor/raça (validação):")
print(va.round(1).to_string())

erro_ind = {nome: va.loc["Indígena", f"erro · {nome} (pontos)"] for nome in VARIANTES} if "Indígena" in va.index else {}
pior = {nome: va[f"erro · {nome} (pontos)"].abs().max() for nome in VARIANTES}
ganho_auc = geral.loc["principal + cor/raça", "AUC"] - geral.loc["principal", "AUC"]
print(f"\nErro com indígenas · principal: {erro_ind.get('principal', np.nan):+.1f} pontos · "
      f"com cor/raça: {erro_ind.get('principal + cor/raça', np.nan):+.1f} pontos")
print(f"Maior erro entre grupos · principal: {pior['principal']:.1f} · com cor/raça: {pior['principal + cor/raça']:.1f}")
print(f"Diferença de AUC geral: {ganho_auc:+.4f}")
melhora = (erro_ind and abs(erro_ind["principal + cor/raça"]) < abs(erro_ind["principal"]) - 2
           and ganho_auc > -0.002)
print("\nCritério: adotar se o erro com indígenas cair mais de 2 pontos sem perder mais de 0,002 de AUC.")
print("→ ADOTAR cor/raça como entrada" if melhora else "→ NÃO há ganho suficiente para adotar")

# %% [markdown]
# ## Teste: só para relatar

# %%
te = por_grupo("teste")
print("Por cor/raça (teste):")
print(te.round(1).to_string())
print({nome: round(roc_auc_score(partes["teste"]["evadiu"], prob[(nome, "teste")]), 4) for nome in VARIANTES})
pd.concat({"validacao": va, "teste": te}).round(3).to_csv(PASTA_RES / "cor_raca_entrada.csv")
