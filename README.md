# Risco de evasão na Rede Federal do Nordeste

Projeto final da disciplina de Inteligência Artificial (UFPI, 2026.2), de Maria Ingrid Xavier Sousa.

Um modelo de classificação estima, a partir de dados conhecidos no ingresso (curso, campus, oferta, idade e sexo), a probabilidade de um aluno de curso técnico ou de graduação evadir. O trabalho avalia o desempenho contra baselines simples, a calibração das probabilidades e a desigualdade de erros entre grupos de alunos.

- **Demonstração:** https://mingridxs.github.io/NOME-DO-REPOSITORIO/ (o modelo roda no navegador; nada é enviado ou guardado)
- **Notebook:** LINK DO COLAB OU KAGGLE
- **Dados limpos:** LINK DO DATASET NO KAGGLE

## Resultados principais (teste, 17.849 matrículas)

| Modelo | AUC (IC 95%) | Brier |
| --- | --- | --- |
| Classe majoritária | 0,500 | 0,243 |
| Taxa histórica do curso no campus | 0,761 (0,755–0,769) | 0,195 |
| Regressão logística | 0,779 (0,772–0,787) | 0,187 |
| **Gradient boosting (modelo principal)** | **0,815 (0,809–0,822)** | **0,172** |
| Gradient boosting com renda e cor/raça | 0,833 (0,828–0,839) | 0,163 |

O modelo principal é bem calibrado para quase todos os grupos, mas tem taxas de erro diferentes entre grupos com taxas de evasão diferentes (sexo, cor/raça, idade). Ele subestima o risco de estudantes indígenas e quase não detecta a evasão de quem entra com até 17 anos. Os detalhes estão no artigo.

## Dados

Microdados de Eficiência Acadêmica da [Plataforma Nilo Peçanha](https://dadosabertos.mec.gov.br/pnp) (MEC), edições 2021, 2022 e 2023, licença [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Os arquivos originais (cerca de 800 MB cada) não estão no repositório: baixe-os nas páginas abaixo e salve-os na raiz do projeto.

- [2021 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/180-2021-microdados-eficiencia-academica) → `microdados_eficiencia_academica_2021.csv`
- [2022 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/254-2022-microdados-eficiencia-academica) → `microdados_eficiencia_academica_2022.csv`
- [2023 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/261-2023-microdados-eficiencia-academica) → `microdados_eficiencia_academica_2023.csv`

A base derivada (`ef_2021_2023.csv`) é distribuída sob a mesma licença ODbL.

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Limpeza de cada edição (recorte Nordeste, cursos técnicos e de graduação)
python preprocessa_pnp_v2.py microdados_eficiencia_academica_2021.csv --ano 2021 --regiao Nordeste --saida ef_2021.csv
python preprocessa_pnp_v2.py microdados_eficiencia_academica_2022.csv --ano 2022 --regiao Nordeste --saida ef_2022.csv
python preprocessa_pnp_v2.py microdados_eficiencia_academica_2023.csv --ano 2023 --regiao Nordeste --saida ef_2023.csv

# 2. Junção e divisão 70/20/10 (estratificada, semente 42)
python junta_edicoes_v2.py

# 3. Análises e modelos
python analise_exploratoria.py     # só treino
python modelagem.py                # baselines, gradient boosting, calibração, teste
python analise_vies.py             # erros por grupo, no teste
python experimentos_melhoria.py    # atributos novos, limiar por nível, curva de aprendizado
python analise_confianca.py        # acerto por nível de confiança

# 4. Site
python exporta_modelo.py           # gera docs/modelo.json e docs/opcoes.json
node verifica_site.mjs             # confere que o JavaScript calcula o mesmo que o Python
```

As versões `_v1` dos dois primeiros scripts ficam no repositório como histórico. A v2 acrescentaria colunas de demanda (inscritos por vaga), mas os arquivos de Eficiência Acadêmica não trazem a coluna de inscritos, então as duas versões geram a mesma base.

Para ver o site localmente: `cd docs && python -m http.server` e abra http://localhost:8000.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `*.py` | Etapas do trabalho, na ordem acima; cada arquivo tem células `# %%` para rodar no VS Code |
| `figuras/` | Gráficos usados no artigo e nos slides |
| `resultados/` | Métricas, busca de hiperparâmetros, análise de viés, previsões do teste |
| `docs/` | Site do GitHub Pages: página, modelo exportado e cálculo em JavaScript |

## Uso de modelos de linguagem

O Claude (Anthropic) foi usado para sugerir o tema e o desenho do experimento, buscar bases e documentação, gerar e depurar o código e organizar a documentação do projeto. Todas as execuções foram feitas pela autora nos dados reais, e as decisões estão registradas na documentação do projeto e no artigo.
