# %% [markdown]
# # 19. Onde o modelo vai pior, e o que dá para fazer
#
# Duas perguntas práticas para a assistência estudantil:
#
# 1. **Se a equipe só consegue atender os K% de maior risco**, quantos deles
#    realmente evadiriam (precisão), e que parte de todas as evasões eles cobrem?
# 2. **Estudantes de até 17 anos** (sobretudo técnico integrado): só 39% das
#    evasões são detectadas. Um limiar próprio para eles, escolhido na validação
#    pela acurácia balanceada, melhora? A que custo em falsos alarmes?
#
# Critério para o limiar por idade, definido antes de rodar (na validação):
# adotar se a detecção do grupo subir pelo menos 10 pontos e a acurácia
# balanceada geral não cair. O teste só é relatado.

# %%
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PASTA_DADOS, PASTA_RES = Path("dados/processados"), Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}
GRADE = np.linspace(0.05, 0.95, 91)



def limiar_por_aluno(d, final):
    """Limiar de alerta de cada aluno: por nível e, se adotado (seção 6.2), por declaração de renda."""
    if final.get("limiar_renda"):
        nd = d["renda_nao_declarada"].astype(int)
        return np.array([final["limiar_renda"][f"{n}|{r}"] for n, r in zip(d["nivel"], nd)])
    lim = final["limiar"]
    return d["nivel"].map(lim).to_numpy() if isinstance(lim, dict) else np.full(len(d), float(lim))


def caminho(pasta, nome):
    if (pasta / nome).exists():
        return pasta / nome
    achados = sorted(glob.glob(f"/kaggle/input/**/{nome}", recursive=True))
    if not achados:
        raise SystemExit(f"Não achei {nome}. Rode o notebook antes ou adicione a saída dele como entrada.")
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
partes = {p: df[df["particao"] == p].reset_index(drop=True) for p in ["validacao", "teste"]}
for d in partes.values():
    d["prob"] = final["calibrador"].predict(final["modelo"].predict_proba(d[CAT + NUM])[:, 1])
    d["jovem"] = d["idade_ingresso"] <= 17

# %% [markdown]
# ## 1. Atender só os K% de maior risco

# %%
te = partes["teste"]
linhas = []
for nivel, d in [("Todos", te)] + list(te.groupby("nivel")):
    ordem = d.sort_values("prob", ascending=False)
    for k in [5, 10, 20, 30]:
        top = ordem.head(int(len(ordem) * k / 100))
        linhas.append({"nível": nivel, "K (%)": k, "alunos atendidos": len(top),
                       "precisão (%)": top["evadiu"].mean() * 100,
                       "evasões cobertas (%)": top["evadiu"].sum() / d["evadiu"].sum() * 100,
                       "evasão média do nível (%)": d["evadiu"].mean() * 100})
topk = pd.DataFrame(linhas)
print("Teste — atender os K% de maior risco:")
print(topk.round(1).to_string(index=False))

# %% [markdown]
# ## 2. Limiar próprio para estudantes de até 17 anos

# %%
def melhor_bal(y, p):
    return float(GRADE[int(np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in GRADE]))])


va = partes["validacao"]
LIM_JOVEM = {n: melhor_bal(va.loc[va["jovem"] & (va["nivel"] == n), "evadiu"],
                           va.loc[va["jovem"] & (va["nivel"] == n), "prob"])
             for n in ["Técnico", "Graduação"] if (va["jovem"] & (va["nivel"] == n)).sum() >= 100}
print("Limiares próprios para até 17 anos:", LIM_JOVEM, "| atuais:", LIMIAR)


def alerta(d, com_jovem):
    lim = limiar_por_aluno(d, final).astype(float)
    if com_jovem:
        for n, t in LIM_JOVEM.items():
            lim[(d["jovem"] & (d["nivel"] == n)).to_numpy()] = t
    return (d["prob"].to_numpy() >= lim).astype(int)


def resumo(d, com_jovem):
    a, y, j = alerta(d, com_jovem), d["evadiu"].to_numpy(), d["jovem"].to_numpy()
    return {"acurácia balanceada (%)": balanced_accuracy_score(y, a) * 100,
            "AUC até 17": roc_auc_score(y[j], d["prob"].to_numpy()[j]),
            "evasão real até 17 (%)": y[j].mean() * 100,
            "detecção até 17 (%)": a[j & (y == 1)].mean() * 100,
            "falsos alarmes até 17 (%)": a[j & (y == 0)].mean() * 100,
            "detecção 18+ (%)": a[~j & (y == 1)].mean() * 100,
            "falsos alarmes 18+ (%)": a[~j & (y == 0)].mean() * 100}


tab = pd.concat({p: pd.DataFrame({"atual": resumo(d, False), "limiar próprio até 17": resumo(d, True)}).T
                 for p, d in partes.items()})
print(tab.round(3).to_string())
v = tab.loc["validacao"]
ok = (v.loc["limiar próprio até 17", "detecção até 17 (%)"] >= v.loc["atual", "detecção até 17 (%)"] + 10
      and v.loc["limiar próprio até 17", "acurácia balanceada (%)"] >= v.loc["atual", "acurácia balanceada (%)"])
print("Decisão:", "ADOTAR limiar próprio para até 17 anos" if ok else "manter limiares atuais")

PASTA_RES.mkdir(exist_ok=True)
topk.to_csv(PASTA_RES / "atendimento_top_k.csv", index=False)
tab.to_csv(PASTA_RES / "limiar_ate_17.csv")
