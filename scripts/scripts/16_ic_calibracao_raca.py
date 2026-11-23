# %% [markdown]
# # 16. Incerteza do erro de calibração por cor/raça
#
# O erro de calibração dos estudantes indígenas mudou de −3,0 pontos na validação
# para −6,6 no teste. Aqui se mede quanto disso pode ser acaso: para cada grupo de
# cor/raça, o erro (risco médio previsto − evasão observada, em pontos) recebe um
# intervalo de confiança de 95% por bootstrap. Nada no modelo muda.

# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SEMENTE, REPETICOES = 42, 2000
PASTA_DADOS, PASTA_RES = Path("dados/processados"), Path("resultados")

final = joblib.load(PASTA_RES / "modelo_final.joblib")
pre = final["modelo"].named_steps["pre"]
CAT, NUM = list(pre.transformers_[0][2]), list(pre.transformers_[1][2])
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}

df = pd.read_csv(PASTA_DADOS / "ef_2021_2023.csv")
numericas = {"carga_horaria", "ano_ingresso", "mes_ingresso", "duracao_prevista_meses", "idade_ingresso",
             "ano_base", "evadiu", "retido", "inscritos", "vagas", "inscritos_por_vaga"}
categoricas = [c for c in df.columns if c not in numericas]
df[categoricas] = df[categoricas].astype(str)
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
df["nivel"] = np.where(df["tipo_de_curso"].isin({"Tecnologia", "Licenciatura", "Bacharelado"}), "Graduação", "Técnico")

rng = np.random.default_rng(SEMENTE)
linhas = []
for particao in ("validacao", "teste"):
    d = df[df["particao"] == particao].reset_index(drop=True)
    p = final["calibrador"].predict(final["modelo"].predict_proba(d[CAT + NUM])[:, 1])
    y = d["evadiu"].to_numpy()
    for grupo, idx in d.groupby("cor_raca").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 30:
            continue
        erro = (p[idx].mean() - y[idx].mean()) * 100
        amostras = rng.choice(idx, size=(REPETICOES, len(idx)), replace=True)
        boot = (p[amostras].mean(axis=1) - y[amostras].mean(axis=1)) * 100
        ic_inf, ic_sup = np.percentile(boot, [2.5, 97.5])
        linhas.append({"partição": particao, "cor/raça": grupo, "n": len(idx),
                       "evasão observada (%)": y[idx].mean() * 100, "risco previsto (%)": p[idx].mean() * 100,
                       "erro (pontos)": erro, "IC 95% inferior": ic_inf, "IC 95% superior": ic_sup,
                       "IC inclui zero": "sim" if ic_inf <= 0 <= ic_sup else "não"})

tabela = pd.DataFrame(linhas).set_index(["partição", "cor/raça"])
print(tabela.round(1).to_string())
tabela.to_csv(PASTA_RES / "ic_calibracao_raca.csv")
