// Calcula a explicação do resultado fora da tela principal, para a página não travar.
import { explicar } from "./modelo.js";

let modelo = null, fundo = null;
self.onmessage = ({ data }) => {
  if (data.tipo === "iniciar") { modelo = data.modelo; fundo = data.fundo; return; }
  if (data.tipo === "explicar") {
    const r = explicar(modelo, data.entrada, fundo, data.grupos);
    self.postMessage({ id: data.id, ...r });
  }
};
