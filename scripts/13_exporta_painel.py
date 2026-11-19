"""
Exporta para o site os números da aba "Em números" e a referência usada para
explicar cada resultado.

Gera docs/painel.json com:
  escopos - para o Nordeste e para cada instituição com pelo menos 300
            matrículas: total, evadidos e a evasão por nível do curso, forma de
            oferta, faixa de idade e edição. Grupos com menos de 30 matrículas
            não são publicados.
  fundo   - 40 perfis de referência para a explicação do resultado. Cada um é
            uma oferta de curso sorteada (com chance proporcional ao número de
            matrículas) com idade, sexo, mês, cor/raça e renda declarada ou não
            sorteados de forma independente, da partição de treino. São perfis
            sintéticos: nenhum corresponde a um aluno real.

Uso, na raiz do repositório, depois do script 08:
    python scripts/13_exporta_painel.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 42
MIN_GRUPO, MIN_ESCOPO, N_FUNDO = 30, 300, 40
AUSENTES = {"Não informado", "Não declarada", "Não declarado", "Não Declarada", "nan"}
CHAVE = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso",
         "tipo_de_oferta", "modalidade_de_ensino", "eixo_tecnologico", "subeixo_tecnologico",
         "turno", "fonte_de_financiamento"]

df = pd.read_csv("dados/processados/ef_2021_2023.csv", dtype={c: str for c in CHAVE + ["sexo", "cor_raca", "renda_familiar"]})
for novo, origem in [("renda_nao_declarada", "renda_familiar"), ("cor_nao_declarada", "cor_raca")]:
    df[novo] = (df[origem].isin(AUSENTES) | df[origem].isna()).astype(int)
df["faixa_idade"] = pd.cut(df["idade_ingresso"], [0, 17, 24, 34, 200],
                           labels=["Até 17 anos", "18 a 24 anos", "25 a 34 anos", "35 anos ou mais"])
df["edicao"] = df["ano_base"].astype(int).astype(str)

# %% Evasão por escopo e dimensão
DIMENSOES = {
    "nivel": ("tipo_de_curso", ["Técnico", "Tecnologia", "Licenciatura", "Bacharelado"]),
    "oferta": ("tipo_de_oferta", None),
    "idade": ("faixa_idade", ["Até 17 anos", "18 a 24 anos", "25 a 34 anos", "35 anos ou mais"]),
    "edicao": ("edicao", ["2021", "2022", "2023"]),
}


def resumo(d):
    saida = {"n": int(len(d)), "ev": int(d["evadiu"].sum()), "dims": {}}
    for nome, (coluna, ordem) in DIMENSOES.items():
        g = d.groupby(coluna, observed=True)["evadiu"].agg(["size", "sum"])
        g = g[g["size"] >= MIN_GRUPO]
        if ordem is None:
            g = g.sort_values("size", ascending=False)
            ordem = list(g.index)
        saida["dims"][nome] = [[str(c), int(g.loc[c, "size"]), int(g.loc[c, "sum"])] for c in ordem if c in g.index]
    return saida


escopos = {"Nordeste": resumo(df)}
for inst, d in df.groupby("instituicao"):
    if len(d) >= MIN_ESCOPO:
        escopos[inst] = resumo(d)

# %% Perfis de referência para a explicação
rng = np.random.default_rng(SEMENTE)
treino = df[df["particao"] == "treino"]
ofertas = (treino.groupby(CHAVE, dropna=False)
                 .agg(n=("evadiu", "size"), carga_horaria=("carga_horaria", "median"),
                      duracao_prevista_meses=("duracao_prevista_meses", "median"))
                 .reset_index())
escolhidas = ofertas.iloc[rng.choice(len(ofertas), size=N_FUNDO, p=ofertas["n"] / ofertas["n"].sum())]


def sorteia(coluna):
    return treino[coluna].iloc[rng.integers(0, len(treino), N_FUNDO)].to_numpy()


fundo = escolhidas[CHAVE + ["carga_horaria", "duracao_prevista_meses"]].reset_index(drop=True).copy()
fundo["idade_ingresso"] = sorteia("idade_ingresso")
fundo["sexo"] = sorteia("sexo")
fundo["mes_ingresso"] = sorteia("mes_ingresso")
fundo["cor_raca"] = sorteia("cor_raca")
fundo["cor_nao_declarada"] = (pd.Series(fundo["cor_raca"]).isin(AUSENTES) | pd.Series(fundo["cor_raca"]).isna()).astype(int)
fundo["renda_nao_declarada"] = sorteia("renda_nao_declarada")
fundo = fundo.astype(object).where(fundo.notna(), None)
for c in ["carga_horaria", "duracao_prevista_meses", "idade_ingresso", "mes_ingresso",
          "cor_nao_declarada", "renda_nao_declarada"]:
    fundo[c] = [None if v is None else float(v) for v in fundo[c]]

painel = {"min_grupo": MIN_GRUPO, "escopos": escopos, "fundo": fundo.to_dict(orient="records")}
destino = Path("docs/painel.json")
destino.write_text(json.dumps(painel, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"{destino}: {len(escopos) - 1} instituições, {N_FUNDO} perfis de referência, "
      f"{destino.stat().st_size / 1e3:.0f} kB")
