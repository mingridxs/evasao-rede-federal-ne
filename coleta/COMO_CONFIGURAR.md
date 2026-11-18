# Como ligar o envio das avaliações

O site funciona sem isto: calcula o risco e mostra a avaliação, mas avisa que o envio não está ligado. Para guardar as avaliações numa planilha do Google (gratuito):

1. Crie uma planilha nova no Google Planilhas, por exemplo "Avaliações do site de evasão".
2. Nela, abra **Extensões → Apps Script**. Apague o que aparecer e cole o conteúdo de `coleta/Code.gs`. Salve.
3. Clique em **Implantar → Nova implantação**. Em "Selecionar tipo", escolha **App da Web**.
   - Executar como: **Eu**
   - Quem pode acessar: **Qualquer pessoa**
4. Clique em **Implantar**, autorize o acesso à planilha e copie o endereço que termina em `/exec`.
5. Cole esse endereço em `docs/coleta.js`:
   ```js
   export const ENDERECO_COLETA = "https://script.google.com/macros/s/.../exec";
   ```
6. Faça commit e push. Em um ou dois minutos o site passa a mostrar a caixa de consentimento e o botão de envio.

Para testar, abra o site, calcule um risco, avalie, marque o consentimento e envie. Uma linha nova deve aparecer na aba "respostas" da planilha.

## O que é guardado

Data e hora do envio, o curso e a forma de entrada informados (inclusive cor ou raça), o resultado do modelo, a nota de 1 a 5 e o comentário opcional. Não há nome, e-mail nem endereço IP. Só é enviado quem marca o consentimento.

## Cuidados antes de usar os dados

- Cor ou raça é dado pessoal sensível pela LGPD. O consentimento do site cita esse dado de forma explícita; não o retire do texto. Guarde a planilha só com acesso seu.

- Numa turma pequena, curso, idade e sexo juntos podem identificar alguém. Não publique as linhas da planilha; publique só contagens e médias.
- Para usar as respostas no artigo ou em outra publicação, combine antes com o professor. Pesquisa com pessoas pode exigir aprovação de um comitê de ética.
- Para desligar o envio, deixe `ENDERECO_COLETA` vazio de novo ou arquive a implantação no Apps Script.
