# %% [markdown]
# # 17. Dá para corrigir a subestimação com estudantes indígenas?
#
# O modelo final subestima o risco de estudantes indígenas (−3,0 pontos na
# validação, −6,6 no teste, com intervalos que encostam no zero). Duas correções
# são testadas contra o modelo final:
#
# - **Recalibração por grupo:** soma ao risco previsto de estudantes indígenas
#   a diferença média entre evasão real e risco previsto, aprendida na validação.
# - **Peso no treino:** retreina o mesmo modelo dando peso 3 ou 10 às matrículas
#   de estudantes indígenas, e recalibra na validação como o modelo final.
#
# **Critérios, definidos antes de rodar:**
#
# - **Peso no treino:** adotar se, na validação, (1) o erro absoluto com indígenas
#   cair mais de 2 pontos, (2) a AUC geral não perder mais de 0,002 e (3) o maior
#   erro absoluto entre os demais grupos de cor/raça não subir mais de 1 ponto.
# - **Recalibração por grupo:** o ajuste é a própria média do erro na validação,
#   então medi-lo nos mesmos alunos daria erro zero por construção (validação
#   cruzada também, porque o erro médio do grupo é um número só). Adotar apenas
#   se o intervalo de 95% do erro na validação não incluir zero, isto é, se houver
#   evidência de um viés sistemático a corrigir.
#
# O teste só é relatado, para todas as variantes.

# %%
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

SEMENTE, REPETICOES = 42, 2000
PASTA_DADOS, PASTA_RES = Path("dados/processados"), Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
GRUPO = "Indígena"
PESOS = [3, 10]




def limiar_por_aluno(d, final):
    """Limiar de alerta de cada aluno: por nível e, se adotado (seção 6.2), por declaração de renda."""
    if final.get("limiar_renda"):
        nd = d["renda_nao_declarada"].astype(int)
        return np.array([final["limiar_renda"][f"{n}|{r}"] for n, r in zip(d["nivel"], nd)])
    lim = final["limiar"]
    return d["nivel"].map(lim).to_numpy() if isinstance(lim, dict) else np.full(len(d), float(lim))


def caminho(pasta, nome):
    """Arquivo gerado pelo notebook; numa sessão nova do Kaggle, procura nas entradas."""
    if (pasta / nome).exists():
        return pasta / nome
    achados = sorted(glob.glob(f"/kaggle/input/**/{nome}", recursive=True))
    if not achados:
        raise SystemExit(f"Não achei {nome}. Rode o notebook antes ou adicione a saída dele como entrada.")
    print(nome, "->", achados[0])
    return Path(achados[0])


final = joblib.load(caminho(PASTA_RES, "modelo_final.joblib"))
pre = final["modelo"].named_steps["pre"]
CAT, NUM = list(pre.transformers_[0][2]), list(pre.transformers_[1][2])
LIMIAR = final["limiar"]
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

df = pd.read_csv(caminho(PASTA_DADOS, "ef_2021_2023.csv"),
                 dtype={c: str for c in set(CAT) | {"cor_raca", "renda_familiar", "tipo_de_curso"}})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
df["nivel"] = np.where(df["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
partes = {p: df[df["particao"] == p].reset_index(drop=True) for p in ["treino", "validacao", "teste"]}
for p, d in partes.items():
    print(f"{p}: {len(d):,} matrículas, {int((d['cor_raca'] == GRUPO).sum())} de estudantes indígenas")

# %% [markdown]
# ## Treino das variantes

# %%
def treina(peso_grupo):
    """Mesmo pipeline e hiperparâmetros do modelo final; peso extra para o grupo no treino."""
    tr = partes["treino"]
    pesos = np.where(tr["cor_raca"] == GRUPO, peso_grupo, 1.0)
    modelo = clone(final["modelo"]).fit(tr[CAT + NUM], tr["evadiu"], modelo__sample_weight=pesos)
    bruto_va = modelo.predict_proba(partes["validacao"][CAT + NUM])[:, 1]
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bruto_va, partes["validacao"]["evadiu"])
    return {p: calib.predict(modelo.predict_proba(partes[p][CAT + NUM])[:, 1]) for p in ["validacao", "teste"]}


prob = {"final": {p: final["calibrador"].predict(final["modelo"].predict_proba(partes[p][CAT + NUM])[:, 1])
                  for p in ["validacao", "teste"]}}
for w in PESOS:
    print(f"Treinando com peso {w} para estudantes indígenas…")
    prob[f"peso {w}"] = treina(w)

# Recalibração por grupo: ajuste aprendido em toda a validação, aplicado ao teste.
va, base_va = partes["validacao"], prob["final"]["validacao"]
y_va, m_va = va["evadiu"].to_numpy(), va["cor_raca"].to_numpy() == GRUPO
AJUSTE = y_va[m_va].mean() - base_va[m_va].mean()
p_va = base_va.copy()
p_va[m_va] = np.clip(p_va[m_va] + AJUSTE, 0, 1)  # só para referência: erro ~0 por construção
te = partes["teste"]
p_te = prob["final"]["teste"].copy()
m_te = te["cor_raca"].to_numpy() == GRUPO
p_te[m_te] = np.clip(p_te[m_te] + AJUSTE, 0, 1)
prob["recalibração por grupo"] = {"validacao": p_va, "teste": p_te}
print(f"Ajuste aprendido na validação: {AJUSTE * 100:+.1f} pontos no risco de estudantes indígenas")

# %% [markdown]
# ## Comparação

# %%
rng = np.random.default_rng(SEMENTE)


def resumo(parte, p, com_ic=False):
    d = partes[parte]
    y, cor = d["evadiu"].to_numpy(), d["cor_raca"].to_numpy()
    lim = limiar_por_aluno(d, final)
    ind = cor == GRUPO
    erros = {g: (p[cor == g].mean() - y[cor == g].mean()) * 100
             for g in pd.unique(cor) if (cor == g).sum() >= 100 and g != GRUPO}
    linha = {"AUC geral": roc_auc_score(y, p),
             "erro indígenas (pontos)": (p[ind].mean() - y[ind].mean()) * 100,
             "evasões de indígenas detectadas (%)": (p[ind & (y == 1)] >= lim[ind & (y == 1)]).mean() * 100,
             "maior erro nos demais grupos (pontos)": max(abs(e) for e in erros.values())}
    if com_ic:
        i = np.flatnonzero(ind)
        amostras = rng.choice(i, size=(REPETICOES, len(i)), replace=True)
        boot = (p[amostras].mean(axis=1) - y[amostras].mean(axis=1)) * 100
        linha["IC 95% erro indígenas"] = "{:+.1f} a {:+.1f}".format(*np.percentile(boot, [2.5, 97.5]))
    return linha


tab_va = pd.DataFrame({nome: resumo("validacao", p["validacao"], com_ic=(nome == "final"))
                       for nome, p in prob.items()}).T
print("Validação (base da decisão; a linha da recalibração tem erro ~0 por construção):")
print(tab_va.round(4).to_string())

b = tab_va.loc["final"]
decisao = {}
inf, sup = (float(x) for x in b["IC 95% erro indígenas"].split(" a "))
decisao["recalibração por grupo"] = not (inf <= 0 <= sup)
print(f"recalibração por grupo: IC do erro na validação {b['IC 95% erro indígenas']} → "
      f"{'ADOTAR' if decisao['recalibração por grupo'] else 'não adotar (sem evidência de viés sistemático)'}")
for nome in [n for n in tab_va.index if n.startswith("peso")]:
    v = tab_va.loc[nome]
    decisao[nome] = (abs(v["erro indígenas (pontos)"]) < abs(b["erro indígenas (pontos)"]) - 2
                     and v["AUC geral"] - b["AUC geral"] >= -0.002
                     and v["maior erro nos demais grupos (pontos)"] <= b["maior erro nos demais grupos (pontos)"] + 1)
    print(f"{nome}: {'ADOTAR' if decisao[nome] else 'não adotar'}")

tab_te = pd.DataFrame({nome: resumo("teste", p["teste"], com_ic=True) for nome, p in prob.items()}).T
print("\nTeste (só para relatar):")
print(tab_te.round(4).to_string())

PASTA_RES.mkdir(exist_ok=True)
pd.concat({"validacao": tab_va, "teste": tab_te}).to_csv(PASTA_RES / "melhora_indigenas.csv")
