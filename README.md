# Risco de evasão na Rede Federal do Nordeste

Projeto final da disciplina de Inteligência Artificial (UFPI, 2026.2), de Maria Ingrid Xavier Sousa.

Um modelo de classificação estima, a partir de dados conhecidos no ingresso (curso, campus, oferta, idade e sexo), a probabilidade de um aluno de curso técnico ou de graduação evadir. O trabalho avalia o desempenho contra baselines simples, a calibração das probabilidades e a desigualdade de erros entre grupos de alunos.

- **Demonstração:** https://mingridxs.github.io/evasao-rede-federal-ne/ (o modelo roda no navegador; só a avaliação opcional, com consentimento, é enviada)
- **Notebook:** em preparação
- **Dados limpos:** em preparação (Kaggle)

## Resultados principais (teste, 17.849 matrículas)

| Modelo | AUC (IC 95%) | Brier |
| --- | --- | --- |
| Classe majoritária | 0,500 | 0,243 |
| Taxa histórica do curso no campus | 0,761 (0,755–0,769) | 0,195 |
| Regressão logística | 0,789 (0,783–0,797) | 0,183 |
| Gradient boosting sem renda e cor/raça | 0,815 (0,809–0,822) | 0,172 |
| Gradient boosting com indicadores de não declaração | 0,833 (0,828–0,839) | 0,163 |
| **Gradient boosting (modelo principal: indicadores e cor/raça)** | **0,832 (0,827–0,838)** | **0,164** |
| Gradient boosting com os valores de renda e cor/raça | 0,833 (0,828–0,839) | 0,163 |

O modelo principal usa curso, campus, oferta, idade, sexo, mês de ingresso e cor/raça, mais dois indicadores: se a renda e se a cor/raça foram registradas na matrícula. Os registros são feitos no ingresso, e quem não os tem evade mais mesmo dentro de cada instituição; o valor da renda não melhora a previsão (`scripts/10_renda_raca.py`) nem a calibração por faixa de renda (`scripts/12_renda_entrada.py`). A cor/raça quase não muda a AUC, mas reduz a subestimação do risco de estudantes indígenas de 10,6 para 5,8 pontos no teste (`scripts/11_cor_raca_entrada.py`).

Com um limiar por nível de curso (0,47 no técnico e 0,77 na graduação), o modelo acerta 73% das matrículas do teste, contra 68% da taxa histórica do curso; nos 30% de casos em que está mais seguro, acerta 86%. Ele é bem calibrado para quase todos os grupos, mas tem taxas de erro diferentes entre grupos com taxas de evasão diferentes (sexo, cor/raça, idade), ainda subestima o risco de estudantes indígenas e encontra só cerca de 40% dos casos de evasão de quem entra com até 17 anos.

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
python scripts/07_analise_confianca.py      # acerto por nível de confiança

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
| `figuras/` | Gráficos usados no artigo e nos slides |
| `resultados/` | Métricas, busca de hiperparâmetros, análise de viés, previsões do teste |
| `docs/` | Site do GitHub Pages: página, modelo exportado e cálculo em JavaScript |
| `coleta/` | Código da planilha do Google que recebe as avaliações do site, e como configurá-la |
| `historico/` | Primeira versão do pré-processamento e da junção. A segunda versão tentou acrescentar a demanda pelo curso (inscritos por vaga), mas os arquivos de Eficiência Acadêmica não trazem a coluna de inscritos; as duas geram a mesma base |
| `referencias/` | Instruções oficiais dos microdados da PNP |
| `dados/` | Microdados e bases derivadas; fora do git |

## Site e avaliação

O site aceita instituições fora da base (por exemplo, universidades federais). Nesse caso, instituição e campus ficam como desconhecidos para o modelo, e o que o formulário não pergunta (área, carga horária, duração) é preenchido com o mais comum em cursos parecidos da base. O resultado é uma extrapolação, e a página avisa isso.

No fim, a pessoa pode dar uma nota de 1 a 5 para o quanto o resultado combina com o que vive no curso. O envio é opcional, exige consentimento e não inclui nome nem e-mail; veja `coleta/COMO_CONFIGURAR.md`.

## Uso de modelos de linguagem

O Claude (Anthropic) foi usado para buscar bases e documentação, sugerir o desenho do experimento, gerar e depurar o código e organizar a documentação do projeto. Todas as execuções foram feitas pela autora nos dados reais, e as decisões estão registradas na documentação do projeto e no artigo.
