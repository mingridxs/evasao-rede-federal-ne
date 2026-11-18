// Recebe as avaliações enviadas pelo site e grava uma linha por envio na aba "respostas".
// Instalação: veja coleta/COMO_CONFIGURAR.md.

const COLUNAS = [
  "recebido_em", "versao_site", "modo", "instituicao", "campus", "uf", "curso", "tipo_de_curso",
  "tipo_de_oferta", "turno", "modalidade", "eixo", "sexo", "idade", "mes",
  "probabilidade", "limiar", "alerta", "avaliacao", "comentario",
];

// Impede que um texto enviado seja interpretado como fórmula pela planilha.
function seguro(valor) {
  if (typeof valor !== "string") return valor;
  return /^[=+\-@]/.test(valor) ? "'" + valor : valor;
}

function doPost(e) {
  const trava = LockService.getScriptLock();
  trava.waitLock(10000);
  try {
    const arquivo = SpreadsheetApp.getActiveSpreadsheet();
    const aba = arquivo.getSheetByName("respostas") || arquivo.insertSheet("respostas");
    if (aba.getLastRow() === 0) aba.appendRow(COLUNAS);
    const dados = JSON.parse(e.postData.contents);
    aba.appendRow(COLUNAS.map(c => c === "recebido_em" ? new Date() : seguro(dados[c] ?? "")));
    return ContentService.createTextOutput("ok");
  } finally {
    trava.releaseLock();
  }
}
