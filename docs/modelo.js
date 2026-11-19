// Cálculo do risco de evasão no navegador.
// Reproduz a previsão do HistGradientBoostingClassifier do scikit-learn
// (mesmas árvores, mesmas regras para categorias e valores ausentes) e a
// calibração isotônica, a partir de docs/modelo.json.

function bitLigado(bitset, valor) {
  return ((bitset[valor >>> 5] >>> (valor & 31)) & 1) === 1;
}

function percorrer(arvore, x, modelo) {
  let i = 0;
  while (!arvore.folha[i]) {
    const v = x[arvore.f[i]];
    const faltante = arvore.m[i] === 1 ? arvore.e[i] : arvore.d[i];
    if (v === null || Number.isNaN(v)) {
      i = faltante;
    } else if (arvore.c[i]) {
      if (v < 0) i = faltante;
      else if (bitLigado(arvore.bits[arvore.b[i]], v)) i = arvore.e[i];
      else if (bitLigado(modelo.conhecidas[modelo.f_idx_map[arvore.f[i]]], v)) i = arvore.d[i];
      else i = faltante; // categoria desconhecida é tratada como ausente
    } else {
      i = v <= arvore.t[i] ? arvore.e[i] : arvore.d[i];
    }
  }
  return arvore.v[i];
}

// Mesma regra do IsotonicRegression.predict com out_of_bounds="clip".
function interpolar(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const meio = (lo + hi) >> 1;
    if (xs[meio] <= x) lo = meio; else hi = meio;
  }
  if (xs[hi] === xs[lo]) return ys[hi];
  return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
}

// Converte a entrada (nomes das colunas -> valores) no vetor que o modelo usa:
// categóricas viram o índice da categoria (ou NaN, se desconhecida), numéricas passam direto.
function vetorizar(modelo, entrada) {
  const x = [];
  for (const c of modelo.categoricas) {
    const valor = entrada[c];
    const idx = valor === null || valor === undefined ? -1 : modelo.categorias[c].indexOf(String(valor));
    x.push(idx < 0 ? NaN : idx);
  }
  for (const c of modelo.numericas) {
    const valor = entrada[c];
    x.push(valor === null || valor === undefined || valor === "" ? NaN : Number(valor));
  }
  return x;
}

function probDoVetor(modelo, x) {
  let bruto = modelo.base;
  for (const arvore of modelo.arvores) bruto += percorrer(arvore, x, modelo);
  const probBruta = 1 / (1 + Math.exp(-bruto));
  const prob = interpolar(modelo.calibracao.x, modelo.calibracao.y, probBruta);
  return { probBruta, prob };
}

export function probabilidade(modelo, entrada) {
  return probDoVetor(modelo, vetorizar(modelo, entrada));
}

// Quanto cada grupo de atributos empurra o risco para cima ou para baixo, em
// relação a perfis de referência (valores de Shapley por grupo, com a
// probabilidade calibrada). Para cada subconjunto S de grupos, v(S) é o risco
// médio quando os grupos de S têm os valores da pessoa e os demais vêm de um
// perfil de referência. As contribuições somam prob(pessoa) − média de v(∅).
// grupos: [{ nome, colunas: [...] }]. Colunas fora do modelo são ignoradas.
export function explicar(modelo, entrada, fundo, grupos) {
  const colunas = [...modelo.categoricas, ...modelo.numericas];
  const usados = grupos.map(g => ({ ...g, idx: g.colunas.map(c => colunas.indexOf(c)).filter(i => i >= 0) }))
                       .filter(g => g.idx.length);
  const k = usados.length;
  const x = vetorizar(modelo, entrada);
  const bs = fundo.map(b => vetorizar(modelo, b));
  const v = new Float64Array(1 << k);
  for (let s = 0; s < 1 << k; s++) {
    let soma = 0;
    for (const b of bs) {
      const z = b.slice();
      for (let g = 0; g < k; g++) if (s >> g & 1) for (const i of usados[g].idx) z[i] = x[i];
      soma += probDoVetor(modelo, z).prob;
    }
    v[s] = soma / bs.length;
  }
  const fat = [1];
  for (let i = 1; i <= k; i++) fat[i] = fat[i - 1] * i;
  const bits = s => { let c = 0; while (s) { c += s & 1; s >>= 1; } return c; };
  const contribuicoes = usados.map((g, i) => {
    let phi = 0;
    for (let s = 0; s < 1 << k; s++) {
      if (s >> i & 1) continue;
      const t = bits(s);
      phi += fat[t] * fat[k - t - 1] / fat[k] * (v[s | 1 << i] - v[s]);
    }
    return { nome: g.nome, valor: phi };
  });
  return { referencia: v[0], prob: v[(1 << k) - 1], contribuicoes };
}

const GRADUACAO = new Set(["Tecnologia", "Licenciatura", "Bacharelado"]);

export function nivelDoCurso(tipoDeCurso) {
  return GRADUACAO.has(tipoDeCurso) ? "Graduação" : "Técnico";
}

export function limiarDoNivel(modelo, tipoDeCurso) {
  return modelo.limiar[nivelDoCurso(tipoDeCurso)];
}
