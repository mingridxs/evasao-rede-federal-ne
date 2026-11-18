// Confere se o cálculo em JavaScript (docs/modelo.js) dá o mesmo resultado
// do Python para 2.000 matrículas do teste.
// Uso: node verifica_site.mjs   (depois de python exporta_modelo.py)
import { readFileSync } from "node:fs";
import { probabilidade } from "./docs/modelo.js";

const modelo = JSON.parse(readFileSync("docs/modelo.json", "utf-8"));
const casos = JSON.parse(readFileSync("resultados/verificacao_site.json", "utf-8"));

let maiorBruta = 0, maior = 0;
for (const caso of casos) {
  const { probBruta, prob } = probabilidade(modelo, caso.entrada);
  maiorBruta = Math.max(maiorBruta, Math.abs(probBruta - caso.prob_bruta));
  maior = Math.max(maior, Math.abs(prob - caso.prob));
}
const ok = maiorBruta < 1e-6 && maior < 1e-6;
console.log(`${casos.length} casos · maior diferença antes da calibração: ${maiorBruta.toExponential(2)}` +
            ` · depois: ${maior.toExponential(2)}`);
console.log(ok ? "OK: o site calcula o mesmo que o Python." : "ERRO: o site diverge do Python.");
process.exit(ok ? 0 : 1);
