"""
Monta notebook/evasao_rede_federal_ne.ipynb a partir dos scripts do repositório.

O notebook é a versão de entrega: um arquivo só, que roda de ponta a ponta no
Kaggle (com o dataset dos microdados anexado) ou localmente, com os brutos em
dados/brutos/. O código vem dos scripts, sem cópia manual: cada bloco "# %%"
vira uma célula, e os blocos "# %% [markdown]" viram texto.

Uso, na raiz do repositório:
    python ferramentas/gera_notebook.py
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SCRIPTS = RAIZ / "scripts"
SAIDA = RAIZ / "notebook" / "evasao_rede_federal_ne.ipynb"

celulas = []


def md(texto):
    celulas.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)})


def code(texto):
    texto = texto.strip("\n")
    if texto.strip():
        celulas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": texto.splitlines(keepends=True)})


def de_comentario(bloco):
    """Converte linhas '# texto' de um bloco markdown em texto."""
    linhas = []
    for l in bloco.splitlines():
        if l.startswith("# "):
            linhas.append(l[2:])
        elif l.strip() == "#":
            linhas.append("")
        elif l.strip():
            linhas.append(l)
    return "\n".join(linhas)


def limpa_codigo(texto):
    texto = re.sub(r"^from padroniza import padroniza_nomes\n", "", texto, flags=re.M)
    return texto


def script_em_celulas(nome, titulo_secao=None, rebaixa=1, envolver_em=None):
    """Cada '# %%' vira uma célula. rebaixa: quantos níveis descer os títulos markdown.
    envolver_em: condição; se dada, o código roda só quando ela é verdadeira."""
    texto = (SCRIPTS / nome).read_text(encoding="utf-8")
    partes = re.split(r"^# %%(.*)$", texto, flags=re.M)
    if titulo_secao:
        md(titulo_secao)
    # partes = [antes, marcador1, bloco1, marcador2, bloco2, ...]
    for marcador, bloco in zip(partes[1::2], partes[2::2]):
        if "[markdown]" in marcador:
            texto_md = de_comentario(bloco)
            texto_md = re.sub(r"^(#+) ", lambda m: "#" * min(6, len(m.group(1)) + rebaixa) + " ", texto_md, flags=re.M)
            md(texto_md)
        else:
            corpo = limpa_codigo(bloco)
            if envolver_em and corpo.strip():
                corpo = f"if {envolver_em}:\n" + "\n".join("    " + l if l.strip() else l for l in corpo.strip("\n").splitlines())
            code(corpo)


def modulo_sem_main(nome, renomear=None):
    """Código de um módulo (sem docstring de abertura nem o bloco __main__)."""
    texto = (RAIZ / nome).read_text(encoding="utf-8")
    texto = texto.split('if __name__ == "__main__":')[0]
    texto = re.sub(r'^"""(.*?)"""\n', "", texto, count=1, flags=re.S)
    if renomear:
        for velho, novo in renomear.items():
            texto = re.sub(rf"\b{velho}\b", novo, texto)
    return texto


# ---------------------------------------------------------------- abertura
md("""
# Previsão de evasão na Rede Federal do Nordeste, com análise de viés

**Maria Ingrid Xavier Sousa** · Projeto final de Inteligência Artificial · Ciência da Computação, UFPI, 2026.2 ·
prof. Pedro Almir

Este notebook reproduz o trabalho inteiro, dos microdados brutos aos resultados do artigo:

1. Dados e histórico do pré-processamento (v1 com Matrículas, v2 com Eficiência Acadêmica)
2. Junção das edições 2021–2023, padronização dos nomes e divisão 70/20/10
3. Análise exploratória (só treino)
4. Modelagem: baselines, gradient boosting, calibração e teste
5. Análise de viés por grupo
6. Experimentos de melhoria e nível de confiança
7. Renda e cor/raça: registro no ingresso, indicadores e cor/raça como entrada
8. Modelo do Nordeste x modelo só do IFPI
9. Retidos: análise de sensibilidade

**Links:** [site](https://mingridxs.github.io/evasao-rede-federal-ne/) ·
[repositório](https://github.com/mingridxs/evasao-rede-federal-ne) ·
dados: Plataforma Nilo Peçanha (MEC), licença ODbL.

**Como rodar no Kaggle:** anexe o dataset com os microdados (*Add Input*) e use *Run All*.
Leva cerca de 20 a 30 minutos na CPU padrão. **Localmente:** coloque os brutos em `dados/brutos/`
com os nomes `microdados_eficiencia_academica_ANO.csv` e `microdados_matriculas_ANO.csv`.
""")

code('''
import glob
import os
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

NO_KAGGLE = Path("/kaggle/input").exists()
if NO_KAGGLE:
    os.chdir("/kaggle/working")
for pasta in ["dados/brutos", "dados/processados", "resultados", "figuras"]:
    Path(pasta).mkdir(parents=True, exist_ok=True)

# No Kaggle, os arquivos do dataset ficam em /kaggle/input/...; aqui eles ganham
# atalhos com os nomes que o código usa.
if NO_KAGGLE:
    for arq in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        nome = Path(arq).name
        if nome.startswith("microdados_") and not Path("dados/brutos", nome).exists():
            os.symlink(arq, Path("dados/brutos", nome))
        elif nome == "ef_2021_2023.csv" and not Path("dados/processados", nome).exists():
            shutil.copy(arq, Path("dados/processados", nome))

EDICOES = [2021, 2022, 2023]
BRUTOS_EF = {ano: Path(f"dados/brutos/microdados_eficiencia_academica_{ano}.csv") for ano in EDICOES}
BRUTOS_MAT = sorted(Path("dados/brutos").glob("microdados_matriculas_*.csv"))
TEM_BRUTOS = all(p.exists() for p in BRUTOS_EF.values())
print("Python", sys.version.split()[0], "· pandas", pd.__version__, "· scikit-learn", sklearn.__version__,
      "· matplotlib", matplotlib.__version__)
print("Brutos de Eficiência Acadêmica:", {a: p.exists() for a, p in BRUTOS_EF.items()})
print("Brutos de Matrículas (v1):", [p.name for p in BRUTOS_MAT] or "nenhum")
if not TEM_BRUTOS:
    assert Path("dados/processados/ef_2021_2023.csv").exists(), \\
        "Sem os brutos e sem a base limpa: anexe o dataset do Kaggle ou coloque os arquivos em dados/brutos/."
    print("Sem os brutos: o pré-processamento é pulado e o notebook usa a base limpa do dataset.")
''')

# ---------------------------------------------------------------- 1. dados
md("""
## 1. Dados e pré-processamento

Microdados da [Plataforma Nilo Peçanha](https://dadosabertos.mec.gov.br/pnp) (PNP), do MEC, sob a
Open Data Commons Open Database License (ODbL). O recorte é o Nordeste, em cursos técnicos e de graduação
(tecnologia, licenciatura e bacharelado).

### 1.1 Histórico: a versão 1 usava o arquivo de Matrículas

A primeira versão do pré-processamento usava os microdados de **Matrículas**, que trazem a situação de
cada aluno **no ano-base**. O problema apareceu nos testes: quem evade cedo sai da base no ano seguinte, então
um arquivo anual mostra sobretudo quem ainda está estudando, e a taxa de evasão por coorte fica distorcida.
Além disso, o arquivo de Matrículas não diz qual foi o desfecho final de quem saiu antes.

O código da v1 fica aqui como registro e roda nos arquivos de Matrículas, se eles estiverem no dataset.
""")
code(modulo_sem_main("historico/preprocessa_pnp_v1.py", renomear={"preprocessa": "preprocessa_v1"}))
code('''
if BRUTOS_MAT:
    v1 = preprocessa_v1({int(p.stem.split("_")[-1]): str(p) for p in BRUTOS_MAT}, regiao="Nordeste")
    print(f"\\nv1 (Matrículas): {len(v1):,} matrículas com desfecho, evasão de {v1['evadiu'].mean():.1%}")
else:
    print("Arquivos de Matrículas não encontrados: a v1 não roda aqui.")
''')
md("""
### 1.2 Versão 2: Eficiência Acadêmica

Os microdados de **Eficiência Acadêmica** trazem, para cada turma cujo prazo previsto terminou no ano anterior
ao da edição, a situação final de cada aluno: concluiu, evadiu ou continua em curso (retido). É exatamente o
desfecho que o modelo precisa prever. Por isso a v2 usa as edições 2021, 2022 e 2023 desse arquivo.

Decisões da v2, registradas na documentação do projeto:
- a região vem da UF (a coluna "Região" da PNP traz outra coisa em algumas edições);
- transferência interna (troca de curso na mesma instituição) não conta como evasão;
- a idade no ingresso é recalculada a partir da idade em 31/12 do ano-base;
- "não declarado" vira categoria própria;
- os **retidos** saem da base principal, porque o desfecho deles ainda não existe. Eles são guardados à parte
  para a análise de sensibilidade da seção 9.
""")
code(modulo_sem_main("scripts/01_preprocessa_pnp.py"))
code('''
if TEM_BRUTOS:
    for ano, caminho in BRUTOS_EF.items():
        print(f"\\n========== Edição {ano} ==========")
        com_retidos = preprocessa({ano: str(caminho)}, regiao="Nordeste", manter_retidos=True)
        com_retidos.to_csv(f"dados/processados/ef_{ano}_com_retidos.csv", index=False)
        # A base principal é a mesma, sem os retidos e sem a coluna que os marca.
        principal = com_retidos[com_retidos["retido"] == 0].drop(columns="retido")
        principal.to_csv(f"dados/processados/ef_{ano}.csv", index=False)
        print(f"Base principal de {ano}: {len(principal):,} matrículas · retidos guardados à parte: "
              f"{int(com_retidos['retido'].sum()):,}")
''')

# ---------------------------------------------------------------- 2. junção
md("""
## 2. Junção das edições, padronização dos nomes e divisão

A edição 2023 traz o nome completo das instituições ("Instituto Federal do Piauí"), e as de 2021 e 2022, a
sigla ("IFPI"). Sem padronizar, o modelo trataria a mesma instituição como duas, e o nome passaria a indicar a
edição. A função abaixo também é usada na seção 9.
""")
code(modulo_sem_main("scripts/padroniza.py"))
script_em_celulas("02_junta_edicoes.py", rebaixa=2, envolver_em="TEM_BRUTOS")

# ---------------------------------------------------------------- 3 a 9
SECOES = [
    ("03_analise_exploratoria.py", "## 3. Análise exploratória"),
    ("04_modelagem.py", "## 4. Modelagem"),
    ("05_analise_vies.py", "## 5. Análise de viés"),
    ("06_experimentos_melhoria.py", "## 6. Experimentos de melhoria"),
    ("07_analise_confianca.py", "### 6.2 Nível de confiança"),
    ("10_renda_raca.py", "## 7. Renda e cor/raça\n\n### 7.1 Registro no ingresso e indicadores de não declaração"),
    ("11_cor_raca_entrada.py", "### 7.2 Cor/raça como entrada"),
    ("12_renda_entrada.py", "### 7.3 Faixa de renda como entrada"),
    ("14_modelo_ifpi.py", "## 8. Modelo do Nordeste ou modelo só do IFPI?"),
]
for nome, titulo in SECOES:
    script_em_celulas(nome, titulo_secao=titulo, rebaixa=2)

md("## 9. Retidos: análise de sensibilidade")
script_em_celulas("15_retidos.py", rebaixa=2, envolver_em="TEM_BRUTOS")

md("""
## Conclusão

- O gradient boosting com indicadores de "não declarou renda" e "não declarou cor/raça" e com a cor/raça como
  entrada é o modelo final: AUC de 0,829 no teste, bem calibrado, com limiar de alerta por nível de curso.
- As diferenças entre grupos aparecem sobretudo nas taxas de erro (recall e falsos positivos), porque as taxas de
  evasão diferem; a calibração é boa para quase todos os grupos, exceto estudantes indígenas, cujo risco ainda é
  subestimado.
- Um modelo só do IFPI não supera o regional, que já funciona calibrado dentro do IFPI.
- Os retidos (20% das matrículas com ciclo encerrado) se parecem mais com quem evade; um modelo de "não concluiu
  no prazo" tem o mesmo desempenho e é um pouco mais equilibrado por renda e nível.

O site que usa o modelo final está em https://mingridxs.github.io/evasao-rede-federal-ne/.
""")

nb = {"cells": celulas, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:03d}"
SAIDA.parent.mkdir(exist_ok=True)
SAIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{SAIDA.relative_to(RAIZ)}: {len(celulas)} células")
