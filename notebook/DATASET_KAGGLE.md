# Texto e passos para o dataset no Kaggle

## Configuração do dataset

- **Título:** Evasão na Rede Federal do Nordeste (PNP 2021–2023)
- **Subtítulo:** Microdados de Eficiência Acadêmica da Plataforma Nilo Peçanha e base limpa para previsão de evasão
- **Licença:** Open Data Commons Open Database License (ODbL). É a mesma licença dos dados originais e a ODbL exige que a redistribuição a mantenha.
- **Visibilidade:** pública
- **Arquivos:**
  - `microdados_eficiencia_academica_2021.csv`, `microdados_eficiencia_academica_2022.csv`, `microdados_eficiencia_academica_2023.csv` (brutos, como baixados da PNP)
  - `microdados_matriculas_ANO.csv` (brutos de Matrículas usados na versão 1 do pré-processamento e no script 10)
  - `ef_2021_2023.csv` (base limpa, gerada pelo script 02)

Os nomes dos arquivos precisam ser exatamente esses: é por eles que o notebook encontra os dados.

## Descrição (cole no campo "About Dataset")

**Evasão na Rede Federal do Nordeste: microdados da Plataforma Nilo Peçanha, edições 2021 a 2023**

Este dataset reúne os dados usados no projeto final de Inteligência Artificial de Maria Ingrid Xavier Sousa (Ciência da Computação, UFPI, 2026.2), que estima o risco de evasão de alunos de cursos técnicos e de graduação dos Institutos Federais e escolas técnicas federais do Nordeste e analisa se o modelo erra igualmente entre grupos de estudantes.

**Fonte.** Microdados de Eficiência Acadêmica e de Matrículas da [Plataforma Nilo Peçanha](https://dadosabertos.mec.gov.br/pnp) (Ministério da Educação), publicados sob a Open Data Commons Open Database License (ODbL). Os arquivos brutos estão aqui sem alteração. Cada edição de Eficiência Acadêmica traz as turmas cujo prazo previsto terminou no ano anterior e a situação final de cada aluno: concluiu, evadiu ou continua em curso.

**Base limpa (`ef_2021_2023.csv`).** 178.484 matrículas do Nordeste, em cursos técnicos, de tecnologia, licenciatura e bacharelado, com desfecho conhecido. Evasão de 58,4%. Gerada pelos scripts 01 e 02 do repositório:
- região derivada da UF;
- transferência interna fora da evasão;
- alunos ainda em curso depois do prazo (retidos) fora da base principal;
- idade no ingresso recalculada;
- nomes de instituições padronizados entre as edições;
- divisão aleatória estratificada 70/20/10 (semente 42), na coluna `particao`.

| Coluna | Descrição |
|---|---|
| `ano_base` | Edição da PNP (2021, 2022 ou 2023) |
| `instituicao`, `unidade_de_ensino`, `uf`, `regiao` | Instituição, campus, estado e região |
| `nome_de_curso`, `tipo_de_curso` | Curso e nível (Técnico, Tecnologia, Licenciatura, Bacharelado) |
| `tipo_de_oferta` | Integrado, Concomitante, Subsequente, PROEJA ou "Não se aplica" (graduação) |
| `modalidade_de_ensino`, `turno` | Presencial ou a distância; turno |
| `eixo_tecnologico`, `subeixo_tecnologico` | Área do curso |
| `fonte_de_financiamento` | Programa associado à matrícula |
| `carga_horaria`, `duracao_prevista_meses` | Carga horária e duração prevista do curso |
| `sexo`, `cor_raca`, `renda_familiar` | Como registrados na matrícula; "Não declarada" é uma categoria |
| `ano_ingresso`, `mes_ingresso`, `idade_ingresso` | Quando o aluno entrou e com que idade |
| `evadiu` | 1 = evadiu; 0 = concluiu |
| `particao` | `treino`, `validacao` ou `teste` |

**Uso.** O notebook do projeto, anexado a este dataset, refaz todo o trabalho, do pré-processamento à análise de viés. Código, site e artigo: https://github.com/mingridxs/evasao-rede-federal-ne

**Cuidados.** Os microdados são anonimizados pela PNP. A previsão de risco serve para orientar apoio estudantil, nunca para negar vagas, bolsas ou direitos.

## Passo a passo

1. Em kaggle.com, entre em **Datasets → New Dataset**.
2. Arraste os arquivos listados acima. Os brutos são grandes, então a subida demora; mantenha a aba aberta.
3. Preencha o título. Escolha a licença **ODbL** (no Kaggle aparece como "Open Database, ODbL" ou "ODC Open Database License") e clique em **Create**.
4. Na página do dataset, cole a descrição acima em **About Dataset** e marque o dataset como **Public**.
5. Para o notebook: **Code → New Notebook → File → Import Notebook** e envie `notebook/evasao_rede_federal_ne.ipynb`.
6. No painel direito, **Add Input** → procure o seu dataset → **Add**.
7. Clique em **Save Version → Save & Run All (Commit)**. Isso roda tudo do zero, em 20 a 30 minutos, e guarda as saídas.
8. Quando a versão terminar, abra o notebook e marque como **Public**. Copie os dois links (dataset e notebook) para o README e para o artigo.

Se alguma célula falhar por versão de biblioteca, ative **Internet** em *Settings* do notebook, rode numa célula nova `!pip install -q "scikit-learn>=1.6" "pandas>=2.2"`, reinicie a sessão e rode de novo.
