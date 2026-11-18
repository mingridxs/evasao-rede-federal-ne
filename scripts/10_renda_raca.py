# %% [markdown]
# # Renda e cor/raça: de onde vem o sinal e se dá para usar
#
# O modelo com renda e cor/raça ganhou 0,018 de AUC, mas há suspeita de que
# parte disso venha do "não declarado" refletir o desfecho (o aluno sai antes
# de a instituição completar o cadastro). Este script testa isso de três jeitos:
#
# 1. **Quando a renda é preenchida.** No arquivo de Matrículas 2023, que tem
#    alunos ainda em curso, compara a falta de renda entre quem acabou de
#    entrar e quem entrou há anos. Se quem acabou de entrar já tem renda, ela
#    é coletada no ingresso e o "não declarado" é informação legítima.
# 2. **Dentro de cada instituição.** Se o "não declarado" evade mais mesmo
#    comparando alunos da mesma instituição, o sinal não é só diferença entre
#    instituições que preenchem mais ou menos.
# 3. **Separar valor e ausência.** Treina variantes do modelo para ver quanto
#    do ganho vem dos valores declarados e quanto vem só de saber quem não
#    declarou. Avaliação na validação; o teste não é usado.
#
# Uso, na raiz do repositório: python scripts/10_renda_raca.py

# %%
import re
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEMENTE = 42
ARQ_MATRICULAS = Path("dados/brutos/microdados_matriculas_2023.csv")
ARQ_BASE = Path("dados/processados/ef_2021_2023.csv")
PASTA_FIG, PASTA_RES = Path("figuras"), Path("resultados")
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
NORDESTE = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}
TIPOS = {"Técnico", "Tecnologia", "Licenciatura", "Bacharelado"}
COR, CINZA = "#2b6cb0", "#9aa5b1"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})


def normaliza(nome):
    sem_acento = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


# %% [markdown]
# ## 1. Quando a renda e a cor/raça são preenchidas (Matrículas 2023)

# %%
if ARQ_MATRICULAS.exists():
    with open(ARQ_MATRICULAS, encoding="latin-1") as f:
        cab = f.readline()
    sep = max([";", "\t", ","], key=cab.count)
    quero = {"categoria_da_situacao", "data_de_inicio_do_ciclo", "renda_familiar", "cor_raca",
             "tipo_de_curso", "uf"}
    for enc in ("utf-8-sig", "latin-1"):
        try:
            m = pd.read_csv(ARQ_MATRICULAS, sep=sep, encoding=enc, dtype=str,
                            usecols=lambda c: normaliza(c) in quero)
            break
        except UnicodeDecodeError:
            continue
    m.columns = [normaliza(c) for c in m.columns]
    m = m[m["uf"].str.upper().isin(NORDESTE) & m["tipo_de_curso"].isin(TIPOS)].copy()
    m["ano_ingresso"] = pd.to_datetime(m["data_de_inicio_do_ciclo"], dayfirst=True, errors="coerce").dt.year
    m["situacao"] = m["categoria_da_situacao"].str.lower().map(
        lambda s: "Em curso" if str(s).startswith("em curso") else
                  "Evadiu" if str(s).startswith("evadid") else "Concluiu")
    m["renda_ausente"] = m["renda_familiar"].isin(AUSENTES) | m["renda_familiar"].isna()
    m["cor_ausente"] = m["cor_raca"].isin(AUSENTES) | m["cor_raca"].isna()
    m = m[m["ano_ingresso"].between(2015, 2023)]
    print(f"Matrículas 2023, Nordeste, técnico e graduação: {len(m):,}")

    tabela = (m.groupby(["ano_ingresso", "situacao"])
                .agg(n=("renda_ausente", "size"), renda_ausente=("renda_ausente", "mean"),
                     cor_ausente=("cor_ausente", "mean"))
                .reset_index())
    tabela = tabela[tabela["n"] >= 200]
    print("\n% sem renda e sem cor/raça declaradas, por ano de ingresso e situação em 2023:")
    print(tabela.assign(renda_ausente=lambda t: (t["renda_ausente"] * 100).round(1),
                        cor_ausente=lambda t: (t["cor_ausente"] * 100).round(1)).to_string(index=False))

    em_curso = tabela[tabela["situacao"] == "Em curso"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(em_curso["ano_ingresso"], em_curso["renda_ausente"] * 100, "o-", color=COR, label="Renda")
    ax.plot(em_curso["ano_ingresso"], em_curso["cor_ausente"] * 100, "o-", color=CINZA, label="Cor/raça")
    ax.set_xlabel("Ano de ingresso (alunos ainda em curso em 2023)")
    ax.set_ylabel("% sem a informação")
    ax.set_ylim(0, 100)
    ax.set_title("Quem acabou de entrar já tem renda e cor/raça registradas?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PASTA_FIG / "50_preenchimento_por_ingresso.png", bbox_inches="tight")
    plt.show()
    print("\nComo ler: se a falta de renda entre os ingressantes de 2023 for parecida com a dos anos\n"
          "anteriores, a renda é registrada no ingresso. Se for muito maior e cair com o tempo de curso,\n"
          "ela é completada depois, e o 'não declarado' carrega informação sobre o desfecho.")
else:
    print(f"{ARQ_MATRICULAS} não encontrado: a parte 1 foi pulada.")

# %% [markdown]
# ## 2. O "não declarado" evade mais dentro da mesma instituição?
#
# Só treino, instituições com pelo menos 300 matrículas e pelo menos 30
# alunos em cada grupo (declarou / não declarou).

# %%
CAT = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso", "tipo_de_oferta",
       "modalidade_de_ensino", "eixo_tecnologico", "subeixo_tecnologico", "turno",
       "fonte_de_financiamento", "sexo"]
EXTRA = ["cor_raca", "renda_familiar"]
NUM = ["carga_horaria", "duracao_prevista_meses", "mes_ingresso", "idade_ingresso"]
df = pd.read_csv(ARQ_BASE, dtype={c: str for c in CAT + EXTRA})
df["renda_ausente"] = df["renda_familiar"].isin(AUSENTES).astype(int)
df["cor_ausente"] = df["cor_raca"].isin(AUSENTES).astype(int)
treino = df[df["particao"] == "treino"].reset_index(drop=True)
validacao = df[df["particao"] == "validacao"].reset_index(drop=True)

linhas = []
for inst, g in treino.groupby("instituicao"):
    for campo in ["renda_ausente", "cor_ausente"]:
        a, d = g[g[campo] == 1], g[g[campo] == 0]
        if len(g) >= 300 and len(a) >= 30 and len(d) >= 30:
            linhas.append({"instituicao": inst, "campo": campo, "n": len(g),
                           "% sem informação": a.shape[0] / len(g) * 100,
                           "evasão sem informação": a["evadiu"].mean() * 100,
                           "evasão com informação": d["evadiu"].mean() * 100})
por_inst = pd.DataFrame(linhas)
por_inst["diferença (pontos)"] = por_inst["evasão sem informação"] - por_inst["evasão com informação"]
for campo, nome in [("renda_ausente", "renda"), ("cor_ausente", "cor/raça")]:
    t = por_inst[por_inst["campo"] == campo]
    if len(t):
        print(f"\n{nome}: {len(t)} instituições comparáveis · em {(t['diferença (pontos)'] > 0).sum()} delas "
              f"quem não declarou evade mais · diferença mediana de {t['diferença (pontos)'].median():.1f} pontos")
        print(f"  % sem {nome} por instituição: de {t['% sem informação'].min():.0f}% a {t['% sem informação'].max():.0f}%")
por_inst.round(1).to_csv(PASTA_RES / "renda_raca_por_instituicao.csv", index=False)

# %% [markdown]
# ## 3. De onde vem o ganho: valores declarados ou ausência?
#
# Variantes, todas com os hiperparâmetros do modelo principal:
#
# - **principal**: sem renda e cor/raça;
# - **+ indicadores de ausência**: principal mais dois campos sim/não (declarou renda? declarou cor/raça?);
# - **completo**: principal mais renda e cor/raça, com "não declarado" como categoria;
# - **declarados**: treinado e avaliado só em quem declarou as duas; compara o principal com o
#   principal mais renda e cor/raça, o que mede o sinal dos valores sem a ausência no meio.

# %%
PARAMS = joblib.load(PASTA_RES / "modelo_principal.joblib")["hiperparametros"]


def pipeline(cats, nums):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cats),
        ("num", "passthrough", nums)])
    return Pipeline([("pre", pre), ("modelo", HistGradientBoostingClassifier(
        categorical_features=list(range(len(cats))), max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, l2_regularization=1.0,
        random_state=SEMENTE, **PARAMS))])


def avaliar(nome, cats, nums, tr, va):
    modelo = pipeline(cats, nums).fit(tr[cats + nums], tr["evadiu"])
    p = modelo.predict_proba(va[cats + nums])[:, 1]
    return {"variante": nome, "matrículas de validação": len(va),
            "AUC": roc_auc_score(va["evadiu"], p), "Brier": brier_score_loss(va["evadiu"], p)}


resultados = [
    avaliar("principal", CAT, NUM, treino, validacao),
    avaliar("+ indicadores de ausência", CAT, NUM + ["renda_ausente", "cor_ausente"], treino, validacao),
    avaliar("completo", CAT + EXTRA, NUM, treino, validacao),
]
decl_tr = treino[(treino["renda_ausente"] == 0) & (treino["cor_ausente"] == 0)]
decl_va = validacao[(validacao["renda_ausente"] == 0) & (validacao["cor_ausente"] == 0)]
resultados += [
    avaliar("declarados: principal", CAT, NUM, decl_tr, decl_va),
    avaliar("declarados: + renda e cor/raça", CAT + EXTRA, NUM, decl_tr, decl_va),
]
tabela = pd.DataFrame(resultados).set_index("variante")
print(tabela.round(4).to_string())
tabela.to_csv(PASTA_RES / "renda_raca_variantes.csv")

ganho_total = tabela.loc["completo", "AUC"] - tabela.loc["principal", "AUC"]
ganho_ausencia = tabela.loc["+ indicadores de ausência", "AUC"] - tabela.loc["principal", "AUC"]
ganho_valores = tabela.loc["declarados: + renda e cor/raça", "AUC"] - tabela.loc["declarados: principal", "AUC"]
print(f"\nGanho do completo sobre o principal: {ganho_total:+.4f}")
print(f"Ganho só com os indicadores de ausência: {ganho_ausencia:+.4f} "
      f"({ganho_ausencia / ganho_total:.0%} do ganho do completo)" if ganho_total > 0 else "")
print(f"Ganho dos valores declarados, entre quem declarou: {ganho_valores:+.4f}")
