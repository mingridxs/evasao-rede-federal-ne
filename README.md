# Risco de evasão na Rede Federal do Nordeste: site de demonstração

Projeto final da disciplina de Inteligência Artificial (UFPI, 2026.2), de Maria Ingrid Xavier Sousa.

Este repositório contém o **site de demonstração** do trabalho, publicado pelo GitHub Pages:
**https://mingridxs.github.io/evasao-rede-federal-ne/**

O site estima, a partir de dados conhecidos na matrícula (curso, campus, oferta, idade, sexo, cor/raça e se renda e cor/raça foram declaradas), o risco de um aluno de curso técnico ou de graduação deixar o curso sem concluir. O modelo (gradient boosting calibrado, AUC de 0,829 no teste) roda no navegador; nada sai do aparelho, a não ser que a pessoa autorize, no fim, o envio da avaliação. É uma demonstração acadêmica, não um diagnóstico.

## Resultados principais (teste, 17.849 matrículas)

| Modelo | AUC (IC 95%) | Brier |
| --- | --- | --- |
| Classe majoritária | 0,500 | 0,243 |
| Taxa histórica do curso no campus | 0,761 (0,754–0,769) | 0,195 |
| Regressão logística | 0,788 (0,782–0,796) | 0,183 |
| Gradient boosting sem renda e cor/raça | 0,814 (0,808–0,820) | 0,172 |
| Gradient boosting com indicadores de não declaração | 0,831 (0,825–0,837) | 0,165 |
| **Gradient boosting (modelo final, usado no site: indicadores e cor/raça)** | **0,829 (0,823–0,835)** | **0,165** |
| Gradient boosting com os valores de renda e cor/raça | 0,830 (0,825–0,837) | 0,164 |

AUC: chance de o modelo dar risco maior a quem evadiu do que a quem concluiu (0,5 é o acaso). Brier: erro médio das probabilidades (menor é melhor).

### Por que o modelo final, e não os de AUC 0,831 ou 0,830

As três versões de gradient boosting empatam: as diferenças de AUC (0,001 a 0,002) ficam dentro dos intervalos de confiança. As escolhas foram feitas na **validação**, nunca pelo resultado do teste.

- **0,831, com os indicadores de não declaração e sem cor/raça:** acrescentar a cor/raça quase não muda a AUC, mas reduz a subestimação do risco de estudantes indígenas, de 5,8 para 3,0 pontos na validação e de 10,4 para 6,6 pontos no teste. Por isso o modelo final usa cor/raça.
- **0,830, com as faixas de renda e a cor/raça declaradas:** na validação, ficou um pouco abaixo do modelo com indicadores (0,8304 contra 0,8315). Quase todo o ganho de renda e cor/raça vem de saber **se** foram declaradas (106% do ganho deste modelo), e, entre quem declarou, a faixa de renda não melhora a previsão. Assim, o site não precisa perguntar a renda de ninguém, só se ela foi informada na matrícula.

O alerta usa limiares por nível de curso e por declaração de renda, escolhidos na validação. Com eles, o modelo acerta 72,1% das matrículas do teste (acurácia balanceada de 72,3%), contra 68,0% da taxa histórica do curso. Nos 30% de casos em que está mais seguro, acerta 83%. Ele é bem calibrado para quase todos os grupos; as limitações (estudantes indígenas, alunos de até 17 anos) estão no notebook e no resumo expandido.

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

O Claude (Anthropic) foi usado para buscar bases e documentação, sugerir o desenho do experimento, gerar e depurar o código e construir o site. Todas as execuções foram feitas pela autora nos dados reais.