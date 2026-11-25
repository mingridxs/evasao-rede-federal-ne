# Risco de evasão na Rede Federal do Nordeste: site de demonstração

Projeto final da disciplina de Inteligência Artificial (UFPI, 2026.2), de Maria Ingrid Xavier Sousa.

Este repositório contém o **site de demonstração** do trabalho, publicado pelo GitHub Pages:
**https://mingridxs.github.io/evasao-rede-federal-ne/**

O site estima, a partir de dados conhecidos na matrícula (curso, campus, oferta, idade, sexo, cor/raça e se renda e cor/raça foram declaradas), o risco de um aluno de curso técnico ou de graduação deixar o curso sem concluir. O modelo (gradient boosting calibrado, AUC de 0,829 no teste) roda no navegador; nada sai do aparelho, a não ser que a pessoa autorize, no fim, o envio da avaliação. É uma demonstração acadêmica, não um diagnóstico.

## Onde está o resto do trabalho

- **Notebook com todo o código das análises**, executado do início ao fim: [Kaggle](https://www.kaggle.com/code/mariaingridxavier/evasao-na-rede-federal-do-ne-previsao-e-vies)
- **Dados** (microdados de Eficiência Acadêmica da Plataforma Nilo Peçanha, 2021 a 2023, e a base limpa; licença ODbL): [Kaggle](https://www.kaggle.com/datasets/mariaingridxavier/evasao-na-rede-federal-do-nordeste-pnp-2021-2023)

## Conteúdo

| Caminho | Conteúdo |
| --- | --- |
| `docs/` | Página, modelo exportado (`modelo.json`), opções de cursos e cálculo do risco em JavaScript |
| `coleta/` | Código da planilha do Google que recebe as avaliações opcionais do site, e como configurá-la |

Para ver o site localmente: `cd docs && python -m http.server` e abra http://localhost:8000.

## Privacidade

O envio da avaliação é opcional, exige consentimento (LGPD) e não inclui nome, e-mail nem endereço de internet. Detalhes no termo exibido no próprio site e em `coleta/COMO_CONFIGURAR.md`.

## Uso de modelos de linguagem

O Claude (Anthropic) foi usado para buscar bases e documentação, sugerir o desenho do experimento, gerar e depurar o código, construir o site e redigir um rascunho inicial do artigo e da sua tradução. Todas as execuções foram feitas pela autora nos dados reais.
