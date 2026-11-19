"""
Figures for the paper (in English), drawn from the CSV files in resultados/.

Run from the repository root, after scripts 04, 07 and 15:
    python artigo/figuras_artigo.py
Writes artigo/fig_models.pdf, artigo/fig_fairness.pdf and artigo/fig_coverage.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("resultados")
OUT = Path("artigo")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "savefig.bbox": "tight"})
BLUE, ORANGE, GREY, INK = "#1f4f8f", "#e08a2e", "#9aa5b1", "#13203a"

# ---------------------------------------------------------------- Figure 1
m = pd.read_csv(RES / "metricas_teste.csv", index_col=0)
rows = [("Majority class", "Classe majoritária"),
        ("Historical course rate", "Taxa histórica do curso"),
        ("Logistic regression", "Regressão logística (principal)"),
        ("GB, no missingness indicators", "Gradient boosting (sem indicadores) · calibrado"),
        ("GB, declared income and race values", "Gradient boosting (completo) · calibrado"),
        ("GB, final model", "Gradient boosting (principal) · calibrado")]
fig, ax = plt.subplots(figsize=(5.6, 1.9))
for i, (label, key) in enumerate(rows):
    auc, lo, hi = m.loc[key, ["AUC", "AUC IC inf", "AUC IC sup"]]
    final = label.endswith("final model")
    ax.barh(i, auc - 0.45, left=0.45, height=0.62, color=BLUE if final else GREY)
    ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="none", ecolor=INK, capsize=2.5, lw=1)
    ax.text(hi + 0.004, i, f"{auc:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0.45, 0.88)
ax.axvline(0.5, color=INK, ls=":", lw=0.8)
ax.set_xlabel("Test AUC (bars: 95% bootstrap CI)")
fig.savefig(OUT / "fig_models.pdf")
fig.savefig(OUT / "fig_models.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
g = pd.read_csv(RES / "retidos_comparacao_grupos.csv")
g = g[g["modelo"] == "Evasão (site)"]
NAMES = {"F": "Female", "M": "Male", "Parda": "Pardo", "Branca": "White", "Preta": "Black", "Amarela": "Asian",
         "Indígena": "Indigenous", "Não declarada": "Race not declared", "declarou": "Income declared",
         "não declarou": "Income not declared", "até 17": "Age ≤17", "18–20": "Age 18–20", "21–24": "Age 21–24",
         "25–29": "Age 25–29", "30 ou mais": "Age ≥30", "Técnico": "Technical", "Graduação": "Undergraduate"}
ORDER = ["Nível", "Sexo", "Cor/raça", "Renda", "Idade no ingresso"]
GROUP_ORDER = ["Técnico", "Graduação", "F", "M", "Branca", "Parda", "Preta", "Amarela", "Indígena", "Não declarada",
               "declarou", "não declarou", "até 17", "18–20", "21–24", "25–29", "30 ou mais"]
g["ordem"] = g["atributo"].map({a: i for i, a in enumerate(ORDER)})
g["ordem_grupo"] = g["grupo"].map({a: i for i, a in enumerate(GROUP_ORDER)})
g = g.sort_values(["ordem", "ordem_grupo"]).reset_index(drop=True)
y = np.arange(len(g))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 4.0), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
a1.hlines(y, g["falsos positivos (%)"], g["recall (%)"], color=GREY, lw=1.2)
a1.scatter(g["recall (%)"], y, color=BLUE, s=22, zorder=3, label="Recall (dropouts flagged)")
a1.scatter(g["falsos positivos (%)"], y, color=ORANGE, s=22, zorder=3, label="False positive rate")
a1.set_yticks(y, [f"{NAMES.get(n, n)} (n={k:,})" for n, k in zip(g["grupo"], g["n"])])
a1.set_xlabel("%")
a1.set_xlim(0, 100)
a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
a1.set_title("Error rates by group")
a2.barh(y, g["erro de calibração (pontos)"], color=GREY, height=0.6)
a2.axvline(0, color=INK, lw=0.8)
a2.set_xlabel("Predicted − observed dropout (points)")
a2.set_title("Calibration error by group")
for ax in (a1, a2):
    for b in np.cumsum(g.groupby("ordem", sort=True).size().values)[:-1]:
        ax.axhline(len(g) - b - 0.5, color=GREY, lw=0.5, ls="--")
fig.tight_layout()
fig.savefig(OUT / "fig_fairness.pdf")
fig.savefig(OUT / "fig_fairness.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
c = pd.read_csv(RES / "acerto_por_cobertura.csv")
fig, ax = plt.subplots(figsize=(4.6, 2.2))
ax.plot(c["cobertura"] * 100, c["acerto"] * 100, marker="o", color=BLUE, ms=4)
for x, v in zip(c["cobertura"] * 100, c["acerto"] * 100):
    ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.5)
ax.axhline(c["acerto"].iloc[-1] * 100, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Share of students decided on, most confident first (%)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(65, 95)
fig.savefig(OUT / "fig_coverage.pdf")
fig.savefig(OUT / "fig_coverage.png", dpi=200)
plt.close(fig)
print("Figures written to artigo/")
