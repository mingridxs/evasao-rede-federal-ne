"""
Exporta o modelo final para o site (GitHub Pages), onde ele roda no navegador.

O GitHub Pages só serve arquivos estáticos, então o modelo vira JSON e o
cálculo é refeito em JavaScript (docs/modelo.js), percorrendo as mesmas
árvores do HistGradientBoostingClassifier e aplicando a mesma calibração.

Gera:
  docs/modelo.json  - árvores, codificação das categorias, calibração e limiares
  docs/opcoes.json  - combinações de campus, curso e oferta para o formulário
  resultados/verificacao_site.json - 2.000 matrículas do teste com a
      probabilidade calculada em Python, para conferir o JavaScript

Uso:  python exporta_modelo.py
Depois: node verifica_site.mjs
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SEMENTE = 42
PASTA_SITE, PASTA_RES = Path("docs"), Path("resultados")
PASTA_SITE.mkdir(exist_ok=True)
MIN_MATRICULAS = 10   # combinações de oferta com menos matrículas não entram no formulário

final = joblib.load(PASTA_RES / "modelo_final.joblib")
pipe, calibrador = final["modelo"], final["calibrador"]
pre, hgb = pipe.named_steps["pre"], pipe.named_steps["modelo"]

cats = list(pre.transformers_[0][2])
nums = list(pre.transformers_[1][2])
encoder = pre.named_transformers_["cat"]
if hgb.n_trees_per_iteration_ != 1:
    raise ValueError("Esperava classificação binária (uma árvore por iteração).")

# %% Árvores
arvores = []
for (pred,) in hgb._predictors:
    n = pred.nodes
    arvores.append({
        "f": n["feature_idx"].astype(int).tolist(),
        "t": [float(x) for x in n["num_threshold"]],
        "e": n["left"].astype(int).tolist(),
        "d": n["right"].astype(int).tolist(),
        "m": n["missing_go_to_left"].astype(int).tolist(),
        "c": n["is_categorical"].astype(int).tolist(),
        "b": n["bitset_idx"].astype(int).tolist(),
        "v": [round(float(x), 10) for x in n["value"]],
        "folha": n["is_leaf"].astype(int).tolist(),
        "bits": pred.raw_left_cat_bitsets.astype(np.uint32).tolist(),
    })
conhecidas, f_idx_map = hgb._bin_mapper.make_known_categories_bitsets()

limiar = final["limiar"]
modelo_json = {
    "versao": "1",
    "variante": final.get("variante", "atual"),
    "categoricas": cats,
    "numericas": nums,
    "categorias": {c: [str(v) for v in cat] for c, cat in zip(cats, encoder.categories_)},
    "base": float(np.ravel(hgb._baseline_prediction)[0]),
    "arvores": arvores,
    "conhecidas": conhecidas.astype(np.uint32).tolist(),
    "f_idx_map": f_idx_map.astype(int).tolist(),
    "calibracao": {"x": calibrador.X_thresholds_.tolist(), "y": calibrador.y_thresholds_.tolist()},
    "limiar": limiar if isinstance(limiar, dict) else {"Técnico": float(limiar), "Graduação": float(limiar)},
}
(PASTA_SITE / "modelo.json").write_text(json.dumps(modelo_json, ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")
print(f"docs/modelo.json: {len(arvores)} árvores, "
      f"{(PASTA_SITE / 'modelo.json').stat().st_size / 1e6:.1f} MB")

# %% Opções do formulário
df = pd.read_csv("ef_2021_2023.csv", dtype={c: str for c in cats})
CHAVE = ["instituicao", "unidade_de_ensino", "uf", "nome_de_curso", "tipo_de_curso",
         "tipo_de_oferta", "modalidade_de_ensino", "eixo_tecnologico", "subeixo_tecnologico",
         "turno", "fonte_de_financiamento"]
ofertas = (df.groupby(CHAVE, dropna=False)
             .agg(n=("evadiu", "size"), carga_horaria=("carga_horaria", "median"),
                  duracao_prevista_meses=("duracao_prevista_meses", "median"))
             .reset_index())
ofertas = ofertas[ofertas["n"] >= MIN_MATRICULAS].drop(columns="n")
for c in ["carga_horaria", "duracao_prevista_meses"]:
    ofertas[c] = ofertas[c].round().astype(int)
meses = df["mes_ingresso"].value_counts(normalize=True)
opcoes = {
    "ofertas": ofertas.to_dict(orient="records"),
    "sexos": sorted(v for v in df["sexo"].dropna().unique() if (df["sexo"] == v).sum() >= 100),
    "meses": sorted(int(m) for m in meses[meses >= 0.02].index),
}
(PASTA_SITE / "opcoes.json").write_text(json.dumps(opcoes, ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")
print(f"docs/opcoes.json: {len(ofertas):,} ofertas de curso (com pelo menos {MIN_MATRICULAS} matrículas)")

# %% Casos para conferir o JavaScript
teste = df[df["particao"] == "teste"]
teste = teste.sample(min(2000, len(teste)), random_state=SEMENTE)
prob_bruta = pipe.predict_proba(teste[cats + nums])[:, 1]
prob = calibrador.predict(prob_bruta)
casos = [{"entrada": {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                      for k, v in linha.items()},
          "prob_bruta": float(pb), "prob": float(p)}
         for linha, pb, p in zip(teste[cats + nums].to_dict(orient="records"), prob_bruta, prob)]
(PASTA_RES / "verificacao_site.json").write_text(json.dumps(casos, ensure_ascii=False), encoding="utf-8")
print(f"resultados/verificacao_site.json: {len(casos):,} casos do teste. Agora rode: node verifica_site.mjs")
