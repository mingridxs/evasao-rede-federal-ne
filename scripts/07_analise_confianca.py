# %% [markdown]
# # Onde o modelo é confiável: acerto por nível de confiança
#
# O acerto geral (cerca de 71%) mistura alunos sobre os quais o modelo tem
# certeza com alunos em que ele está em dúvida. Aqui o acerto é medido em
# função da confiança, no **teste**, com o modelo e os limiares finais
# (limiar por nível de curso). Nada é ajustado.
#
# Confiança de uma previsão = distância entre a probabilidade e o limiar do
# nível do aluno. Quanto mais longe do limiar, mais "decidido" o modelo está.

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PASTA_FIG, PASTA_RES = Path("figuras"), Path("resultados")
COR, CINZA = "#2b6cb0", "#9aa5b1"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
GRADUACAO = {"Tecnologia", "Licenciatura", "Bacharelado"}

final = joblib.load(PASTA_RES / "modelo_final.joblib")
limiares = final["limiar"]
p = pd.read_csv(PASTA_RES / "predicoes_teste.csv")
p["nivel"] = np.where(p["tipo_de_curso"].isin(GRADUACAO), "Graduação", "Técnico")
p["limiar"] = p["nivel"].map(limiares) if isinstance(limiares, dict) else float(limiares)
p["alerta"] = (p["prob_principal"] >= p["limiar"]).astype(int)
p["acertou"] = (p["alerta"] == p["evadiu"]).astype(int)
p["confianca"] = (p["prob_principal"] - p["limiar"]).abs()
print(f"Limiares: {limiares}")
print(f"Acerto geral no teste: {p['acertou'].mean():.1%} em {len(p):,} matrículas")

# %% [markdown]
# ## 1. Acerto por faixa de probabilidade
#
# Para cada faixa: quantos alunos caem nela, quanto evadiram de fato e
# quanto o modelo acertou.

# %%
faixas = pd.cut(p["prob_principal"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True,
                labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
por_faixa = (p.groupby(faixas, observed=True)
               .agg(alunos=("evadiu", "size"), evasao_real=("evadiu", "mean"),
                    acerto=("acertou", "mean"))
               .assign(proporcao=lambda t: t["alunos"] / len(p)))
print((por_faixa * [1, 100, 100, 100]).round(1).rename(columns={
    "evasao_real": "evasão real (%)", "acerto": "acerto (%)", "proporcao": "% dos alunos"}).to_string())

# %% [markdown]
# ## 2. Acerto contra cobertura
#
# Ordena os alunos do mais confiante para o menos confiante. Para cada
# cobertura (fração dos alunos sobre os quais o modelo "decide"), mostra o
# acerto nesse grupo. Os demais ficariam para avaliação humana.

# %%
ordem = p.sort_values("confianca", ascending=False).reset_index(drop=True)
coberturas = np.arange(0.1, 1.01, 0.1)
curva = pd.DataFrame({
    "cobertura": coberturas,
    "alunos": [int(round(c * len(ordem))) for c in coberturas],
    "acerto": [ordem["acertou"].iloc[:int(round(c * len(ordem)))].mean() for c in coberturas],
    "confiança mínima": [ordem["confianca"].iloc[int(round(c * len(ordem))) - 1] for c in coberturas],
})
print((curva.assign(cobertura=curva["cobertura"] * 100, acerto=curva["acerto"] * 100)
            .round(3).to_string(index=False)))
curva.to_csv(PASTA_RES / "acerto_por_cobertura.csv", index=False)

for alvo in [0.80, 0.85, 0.90]:
    ok = curva[curva["acerto"] >= alvo]
    if len(ok):
        print(f"Acerto ≥ {alvo:.0%}: o modelo pode decidir sobre {ok['cobertura'].max():.0%} dos alunos")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curva["cobertura"] * 100, curva["acerto"] * 100, "o-", color=COR)
for _, l in curva.iterrows():
    ax.annotate(f"{l['acerto']:.0%}", (l["cobertura"] * 100, l["acerto"] * 100),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.axhline(p["acertou"].mean() * 100, ls="--", lw=1, color=CINZA)
ax.text(12, p["acertou"].mean() * 100 - 2.5, f"acerto com todos os alunos: {p['acertou'].mean():.0%}",
        fontsize=8, color="#555")
ax.set_xlim(5, 105)
ax.set_ylim(60, 100)
ax.set_xlabel("% dos alunos sobre os quais o modelo decide (os mais confiantes primeiro)")
ax.set_ylabel("Acerto nesse grupo (%)")
ax.set_title("Quanto mais confiante, mais o modelo acerta")
fig.tight_layout()
fig.savefig(PASTA_FIG / "41_acerto_por_cobertura.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Quem fica na zona de dúvida
#
# Os 30% menos confiantes: de que nível e faixa etária são.

# %%
duvida = ordem.iloc[int(0.7 * len(ordem)):]
comparacao = pd.DataFrame({
    "todos (%)": p["nivel"].value_counts(normalize=True) * 100,
    "zona de dúvida (%)": duvida["nivel"].value_counts(normalize=True) * 100}).round(1)
print(comparacao.to_string())
print(f"\nIdade mediana no ingresso · todos: {p['idade_ingresso'].median():.0f} · "
      f"zona de dúvida: {duvida['idade_ingresso'].median():.0f}")
