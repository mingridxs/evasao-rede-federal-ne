# Risco de evasão na Rede Federal do Nordeste

Projeto final da disciplina de Inteligência Artificial (UFPI, 2026.2), de Maria Ingrid Xavier Sousa.

Um modelo de classificação estima, a partir de dados conhecidos no ingresso (curso, campus, oferta, idade, sexo, cor/raça e se renda e cor/raça foram declaradas), a probabilidade de um aluno de curso técnico ou de graduação evadir. O trabalho avalia o desempenho contra baselines simples, a calibração das probabilidades e a desigualdade de erros entre grupos de alunos.

- **Artigo (resumo expandido, SBC):** `artigo/artigo.tex`

- **Demonstração:** https://mingridxs.github.io/evasao-rede-federal-ne/ (o modelo roda no navegador; só a avaliação opcional, com consentimento, é enviada)
- **Notebook:** [no Kaggle](https://www.kaggle.com/code/mariaingridxavier/evasao-na-rede-federal-do-ne-previsao-e-vies), executado do início ao fim com os dados do dataset (cópia em `notebook/evasao_rede_federal_ne.ipynb`)
- **Dados (brutos e limpos):** [Evasao na Rede Federal do Nordeste (PNP 2021-2023)](https://www.kaggle.com/datasets/mariaingridxavier/evasao-na-rede-federal-do-nordeste-pnp-2021-2023), no Kaggle

## Resultados principais (teste, 17.849 matrículas)

| Modelo | AUC (IC 95%) | Brier |
| --- | --- | --- |
| Classe majoritária | 0,500 | 0,243 |
| Taxa histórica do curso no campus | 0,761 (0,754–0,769) | 0,195 |
| Regressão logística | 0,788 (0,782–0,796) | 0,183 |
| Gradient boosting sem renda e cor/raça | 0,814 (0,808–0,820) | 0,172 |
| Gradient boosting com indicadores de não declaração | 0,831 (0,825–0,837) | 0,165 |
| **Gradient boosting (modelo principal: indicadores e cor/raça)** | **0,829 (0,823–0,835)** | **0,165** |
| Gradient boosting com os valores de renda e cor/raça | 0,830 (0,825–0,837) | 0,164 |

O modelo principal usa curso, campus, oferta, idade, sexo, mês de ingresso e cor/raça, mais dois indicadores: se a renda e se a cor/raça foram registradas na matrícula. Os registros são feitos no ingresso, e quem não os tem evade mais mesmo dentro de cada instituição; o valor da renda não melhora a previsão (`scripts/10_renda_raca.py`) nem a calibração por faixa de renda (`scripts/12_renda_entrada.py`). A cor/raça quase não muda a AUC, mas reduz a subestimação do risco de estudantes indígenas de 5,8 para 3,0 pontos na validação e de 10,4 para 6,6 pontos no teste (`scripts/11_cor_raca_entrada.py`).

O alerta usa limiares por nível de curso e por declaração de renda, escolhidos na validação (técnico 0,46 com renda declarada e 0,61 sem; graduação 0,73 e 0,85). Com um limiar só por nível, quem não declarou renda recebia quase o dobro de alarmes falsos (40% contra 21%); com os limiares por renda, a diferença caiu para 4 pontos, e a acurácia balanceada passou de 72,4% para 72,3% (`scripts/18_alarmes_renda.py`). Com essa regra, o modelo acerta 72,1% das matrículas do teste, contra 68,0% da taxa histórica do curso; nos 30% de casos em que está mais seguro, acerta 83%. Ele é bem calibrado para quase todos os grupos, mas ainda subestima o risco de estudantes indígenas (estimativa incerta: só 98 no teste) e encontra só cerca de um terço dos casos de evasão de quem entra com até 17 anos.

Os nomes das instituições mudam entre edições da Plataforma Nilo Peçanha (siglas em 2021 e 2022, nomes completos em 2023) e são padronizados em `scripts/02_junta_edicoes.py`. Antes da padronização, a AUC de teste era 0,832; a queda para 0,829 provavelmente vem de o nome da instituição deixar de indicar a edição, informação que não existe na hora de prever.

## Dados

Microdados de Eficiência Acadêmica da [Plataforma Nilo Peçanha](https://dadosabertos.mec.gov.br/pnp) (MEC), edições 2021, 2022 e 2023, licença [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/). Os arquivos originais (cerca de 800 MB cada) não estão no repositório: baixe-os nas páginas abaixo e salve-os em `dados/brutos/`.

- [2021 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/180-2021-microdados-eficiencia-academica) → `dados/brutos/microdados_eficiencia_academica_2021.csv`
- [2022 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/254-2022-microdados-eficiencia-academica) → `dados/brutos/microdados_eficiencia_academica_2022.csv`
- [2023 – Microdados Eficiência Acadêmica](https://dadosabertos.mec.gov.br/pnp/item/261-2023-microdados-eficiencia-academica) → `dados/brutos/microdados_eficiencia_academica_2023.csv`

A base derivada (`ef_2021_2023.csv`) é distribuída sob a mesma licença ODbL.

## Como reproduzir

Todos os comandos rodam a partir da raiz do repositório.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Limpeza de cada edição (recorte Nordeste, cursos técnicos e de graduação)
for ano in 2021 2022 2023; do
  python scripts/01_preprocessa_pnp.py dados/brutos/microdados_eficiencia_academica_$ano.csv \
      --ano $ano --regiao Nordeste --saida dados/processados/ef_$ano.csv
done

# 2. Junção e divisão 70/20/10 (estratificada, semente 42)
python scripts/02_junta_edicoes.py

# 3. Análises e modelos
python scripts/03_analise_exploratoria.py   # só treino
python scripts/04_modelagem.py              # baselines, gradient boosting, calibração, teste
python scripts/05_analise_vies.py           # erros por grupo, no teste
python scripts/06_experimentos_melhoria.py  # atributo novo, limiar por nível, curva de aprendizado
python scripts/18_alarmes_renda.py          # limiar por nível e por renda (regra adotada)
python scripts/07_analise_confianca.py      # acerto por nível de confiança
# Demais análises (renda e cor/raça, IFPI, retidos, indígenas, pontos fracos): scripts 10 a 19,
# na ordem do notebook (ferramentas/gera_notebook.py). O jeito mais simples de reproduzir tudo é rodar o notebook.

# 4. Site
python scripts/08_exporta_modelo.py         # gera docs/modelo.json e docs/opcoes.json
node scripts/09_verifica_site.mjs           # confere que o JavaScript calcula o mesmo que o Python
python scripts/13_exporta_painel.py         # gera docs/painel.json (aba "Em números" e explicação do resultado)
```

Para ver o site localmente: `cd docs && python -m http.server` e abra http://localhost:8000.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `scripts/` | Etapas do trabalho, numeradas na ordem em que rodam; cada uma tem células `# %%` para rodar no VS Code |
| `notebook/` | Cópia do notebook do Kaggle, gerada a partir dos scripts por `ferramentas/gera_notebook.py` |
| `artigo/` | Resumo expandido (LaTeX, formato SBC), referências e figuras |
| `docs/` | Site do GitHub Pages: página, modelo exportado e cálculo em JavaScript |
| `coleta/` | Código da planilha do Google que recebe as avaliações do site, e como configurá-la |
| `historico/` | Primeira versão do pré-processamento e da junção. A segunda versão tentou acrescentar a demanda pelo curso (inscritos por vaga), mas os arquivos de Eficiência Acadêmica não trazem a coluna de inscritos; as duas geram a mesma base |
| `dados/`, `resultados/`, `figuras/` | Dados, métricas e gráficos gerados ao rodar os scripts; fora do git (os da versão final estão na aba Output do notebook no Kaggle) |

## Site e avaliação

O site aceita instituições fora da base (por exemplo, universidades federais). Nesse caso, instituição e campus ficam como desconhecidos para o modelo, e o que o formulário não pergunta (área, carga horária, duração) é preenchido com o mais comum em cursos parecidos da base. O resultado é uma extrapolação, e a página avisa isso.

O site é uma demonstração acadêmica, não um diagnóstico. No fim, a pessoa pode informar sua situação no curso (cursando, concluiu, saiu sem concluir ou trancou), o ano de ingresso e uma nota de 1 a 5 para o quanto o resultado combina com o que vive no curso. O envio é opcional, exige consentimento e não inclui nome nem e-mail; veja `coleta/COMO_CONFIGURAR.md`.

## Uso de modelos de linguagem

O Claude (Anthropic) foi usado para buscar bases e documentação, sugerir o desenho do experimento, gerar e depurar o código, construir o site e redigir um rascunho inicial do artigo e da sua tradução. Todas as execuções foram feitas pela autora nos dados reais, e as decisões estão registradas na documentação do projeto e no artigo.
