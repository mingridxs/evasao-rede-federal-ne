# %% [markdown]
# # 18. Dá para reduzir os falsos alarmes de quem não declara a renda?
#
# No teste, o modelo final alerta 40,3% dos estudantes que NÃO declararam renda
# e concluíram o curso (falsos alarmes), contra 21,3% de quem declarou. Três
# alternativas são comparadas com o modelo final:
#
# - **Sem indicadores:** o mesmo modelo, retreinado sem os dois indicadores de
#   não declaração (renda e cor/raça), com calibração e limiares por nível
#   escolhidos na validação como no final.
# - **Limiar por nível e renda:** o modelo final com quatro limiares (nível ×
#   declarou renda?), cada um maximizando a acurácia balanceada na validação.
# - **Falsos alarmes iguais:** o modelo final; para quem não declarou renda, o
#   limiar de cada nível é o menor que deixa a taxa de falsos alarmes na
#   validação no máximo igual à de quem declarou.
#
# **Critério, definido antes de rodar (na validação):** adotar a alternativa se
# (1) a diferença de falsos alarmes entre os grupos de renda cair pelo menos
# 5 pontos, (2) a acurácia balanceada geral não cair mais de 1 ponto e
# (3) a detecção de evasões de quem não declarou não ficar abaixo da de quem
# declarou. Se mais de uma passar, fica a de maior acurácia balanceada.
# O teste só é relatado.

# %%
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PASTA_DADOS, PASTA_RES = Path("dados/processados"), Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}
INDICADORES = ["renda_nao_declarada", "cor_nao_declarada"]
GRADE = np.linspace(0.05, 0.95, 91)
NIVEIS = ["Técnico", "Graduação"]


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

df = pd.read_csv(caminho(PASTA_DADOS, "ef_2021_2023.csv"),
                 dtype={c: str for c in set(CAT) | {"cor_raca", "renda_familiar", "tipo_de_curso"}})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
df["nivel"] = np.where(df["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
partes = {p: df[df["particao"] == p].reset_index(drop=True) for p in ["treino", "validacao", "teste"]}

# %% [markdown]
# ## Probabilidades das variantes

# %%
prob = {"final": {p: final["calibrador"].predict(final["modelo"].predict_proba(partes[p][CAT + NUM])[:, 1])
                  for p in ["validacao", "teste"]}}

NUM_SEM = [c for c in NUM if c not in INDICADORES]
sem = clone(final["modelo"])
sem.set_params(pre__transformers=[pre.transformers[0], ("num", "passthrough", NUM_SEM)])
print("Treinando o modelo sem indicadores…")
sem.fit(partes["treino"][CAT + NUM_SEM], partes["treino"]["evadiu"])
bruto_va = sem.predict_proba(partes["validacao"][CAT + NUM_SEM])[:, 1]
calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bruto_va, partes["validacao"]["evadiu"])
prob["sem indicadores"] = {p: calib.predict(sem.predict_proba(partes[p][CAT + NUM_SEM])[:, 1])
                           for p in ["validacao", "teste"]}

# %% [markdown]
# ## Limiares (sempre escolhidos na validação)

# %%
va = partes["validacao"]
y_va = va["evadiu"].to_numpy()
nd_va = va["renda_nao_declarada"].to_numpy() == 1


def melhor_bal(y, p):
    return float(GRADE[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in GRADE]))])


def fpr(y, alerta):
    return alerta[y == 0].mean()


limiares = {}
# Final: limiares do próprio modelo final, por nível.
limiares["final"] = {(n, r): LIMIAR[n] for n in NIVEIS for r in [0, 1]}
# Sem indicadores: limiar por nível, como no final.
p = prob["sem indicadores"]["validacao"]
lim_sem = {n: melhor_bal(y_va[va["nivel"] == n], p[va["nivel"] == n]) for n in NIVEIS}
limiares["sem indicadores"] = {(n, r): lim_sem[n] for n in NIVEIS for r in [0, 1]}
# Limiar por nível e renda.
p = prob["final"]["validacao"]
limiares["limiar por nível e renda"] = {
    (n, r): melhor_bal(y_va[(va["nivel"] == n) & (nd_va == r)], p[(va["nivel"] == n) & (nd_va == r)])
    for n in NIVEIS for r in [0, 1]}
# Falsos alarmes iguais.
lim_ig = {}
for n in NIVEIS:
    m_dec, m_nd = (va["nivel"] == n).to_numpy() & ~nd_va, (va["nivel"] == n).to_numpy() & nd_va
    alvo = fpr(y_va[m_dec], p[m_dec] >= LIMIAR[n])
    lim_ig[(n, 0)] = LIMIAR[n]
    lim_ig[(n, 1)] = next((float(t) for t in GRADE
                           if t >= LIMIAR[n] and fpr(y_va[m_nd], p[m_nd] >= t) <= alvo), 0.95)
limiares["falsos alarmes iguais"] = lim_ig
prob["limiar por nível e renda"] = prob["final"]
prob["falsos alarmes iguais"] = prob["final"]

print("Limiares (nível, 0 = declarou renda, 1 = não declarou):")
for nome, l in limiares.items():
    print(f"  {nome}: " + ", ".join(f"{n}/{r}: {t:.2f}" for (n, r), t in l.items()))

# %% [markdown]
# ## Comparação

# %%
def resumo(parte, nome):
    d, pp = partes[parte], prob[nome][parte]
    y, nd = d["evadiu"].to_numpy(), d["renda_nao_declarada"].to_numpy()
    lim = np.array([limiares[nome][(n, r)] for n, r in zip(d["nivel"], nd)])
    a = (pp >= lim).astype(int)
    linha = {"AUC": roc_auc_score(y, pp), "acurácia balanceada": balanced_accuracy_score(y, a) * 100}
    for r, rot in [(0, "declarou"), (1, "não declarou")]:
        m = nd == r
        linha[f"evasão real ({rot})"] = y[m].mean() * 100
        linha[f"alertas ({rot})"] = a[m].mean() * 100
        linha[f"detecção ({rot})"] = a[m & (y == 1)].mean() * 100
        linha[f"falsos alarmes ({rot})"] = a[m & (y == 0)].mean() * 100
    linha["diferença de falsos alarmes"] = linha["falsos alarmes (não declarou)"] - linha["falsos alarmes (declarou)"]
    return linha


VARIANTES = ["final", "sem indicadores", "limiar por nível e renda", "falsos alarmes iguais"]
tab = {parte: pd.DataFrame({n: resumo(parte, n) for n in VARIANTES}).T for parte in ["validacao", "teste"]}
pd.set_option("display.width", 250)
print("Validação (base da decisão):")
print(tab["validacao"].round(3).to_string())

b = tab["validacao"].loc["final"]
aprovadas = []
for n in VARIANTES[1:]:
    v = tab["validacao"].loc[n]
    ok = (v["diferença de falsos alarmes"] <= b["diferença de falsos alarmes"] - 5
          and v["acurácia balanceada"] >= b["acurácia balanceada"] - 1
          and v["detecção (não declarou)"] >= v["detecção (declarou)"])
    print(f"{n}: {'passa' if ok else 'não passa'} no critério")
    if ok:
        aprovadas.append(n)
escolha = max(aprovadas, key=lambda n: tab["validacao"].loc[n, "acurácia balanceada"]) if aprovadas else "final"
print(f"\nDecisão: {escolha}")

print("\nTeste (só para relatar):")
print(tab["teste"].round(3).to_string())

PASTA_RES.mkdir(exist_ok=True)
pd.concat(tab).to_csv(PASTA_RES / "alarmes_renda.csv")

# %% [markdown]
# ## Regra adotada
#
# Se a alternativa escolhida for "limiar por nível e renda", os quatro limiares
# passam a fazer parte do modelo final e são usados daqui em diante (seções
# seguintes do notebook e site). O modelo e as probabilidades não mudam.

# %%
if escolha == "limiar por nível e renda":
    final["limiar_renda"] = {f"{n}|{r}": round(t, 2) for (n, r), t in limiares[escolha].items()}
    final["regra_limiar"] = "por nível e renda"
    joblib.dump(final, PASTA_RES / "modelo_final.joblib")
    print("Limiares por nível e renda salvos em resultados/modelo_final.joblib:", final["limiar_renda"])
else:
    print("Modelo final mantido com limiares por nível.")
