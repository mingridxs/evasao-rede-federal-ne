# Como ligar o envio das avaliações

O site funciona sem isto: calcula o risco e mostra a avaliação, mas avisa que o envio não está ligado. Para guardar as avaliações numa planilha do Google (gratuito):

1. Crie uma planilha nova no Google Planilhas, por exemplo "Avaliações do site de evasão".
2. Nela, abra **Extensões → Apps Script**. Apague o que aparecer e cole o conteúdo de `coleta/Code.gs`. Salve.
3. Clique em **Implantar → Nova implantação**. Em "Selecionar tipo", escolha **App da Web**.
   - Executar como: **Eu**
   - Quem pode acessar: **Qualquer pessoa**
4. Clique em **Implantar**, autorize o acesso à planilha e copie o endereço que termina em `/exec`.
5. Cole esse endereço em `docs/coleta.js` e, na mesma linha de baixo, um e-mail de contato para pedidos de exclusão (a LGPD exige um canal):
   ```js
   export const ENDERECO_COLETA = "https://script.google.com/macros/s/.../exec";
   export const CONTATO = "seu.email@exemplo.com";
   ```
6. Faça commit e push. Em um ou dois minutos o site passa a mostrar a caixa de consentimento e o botão de envio.

Se a planilha já existia de uma versão anterior do site, cole o novo `Code.gs` (ele completa o cabeçalho da aba "respostas" com as colunas novas no próximo envio, sem apagar as respostas antigas) e publique uma nova versão em **Implantar → Gerenciar implantações → Editar → Nova versão**. Assim o endereço `/exec` continua o mesmo.

Para testar, abra o site, calcule um risco, avalie, marque o consentimento e envie. Uma linha nova deve aparecer na aba "respostas" da planilha.

## O que é guardado

Data e hora do envio, um código aleatório da resposta, o curso e a forma de entrada informados, o resultado do modelo, a nota de 1 a 5 e o comentário opcional. A cor ou raça só é enviada com uma segunda autorização, específica. Não há nome, e-mail nem endereço IP. Só é enviado quem marca o consentimento.

Para atender um pedido de exclusão, procure a linha pelo código que a pessoa informar e apague-a.

## Cuidados antes de usar os dados

- Cor ou raça é dado pessoal sensível pela LGPD. O consentimento do site cita esse dado de forma explícita; não o retire do texto. Guarde a planilha só com acesso seu.

- Numa turma pequena, curso, idade e sexo juntos podem identificar alguém. Não publique as linhas da planilha; publique só contagens e médias.
- Para usar as respostas no artigo ou em outra publicação, combine antes com o professor. Pesquisa com pessoas pode exigir aprovação de um comitê de ética.
- Para desligar o envio, deixe `ENDERECO_COLETA` vazio de novo ou arquive a implantação no Apps Script.
