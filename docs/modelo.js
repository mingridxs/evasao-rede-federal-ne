// Cálculo do risco de evasão no navegador.
// Reproduz a previsão do HistGradientBoostingClassifier do scikit-learn
// (mesmas árvores, mesmas regras para categorias e valores ausentes) e a
// calibração isotônica, a partir de docs/modelo.json.

function bitLigado(bitset, valor) {
  return ((bitset[valor >>> 5] >>> (valor & 31)) & 1) === 1;
}

// Para qual filho o nó i manda o valor v.
function passo(arvore, i, v, modelo) {
  const faltante = arvore.m[i] === 1 ? arvore.e[i] : arvore.d[i];
  if (v === null || Number.isNaN(v)) return faltante;
  if (arvore.c[i]) {
    if (v < 0) return faltante;
    if (bitLigado(arvore.bits[arvore.b[i]], v)) return arvore.e[i];
    if (bitLigado(modelo.conhecidas[modelo.f_idx_map[arvore.f[i]]], v)) return arvore.d[i];
    return faltante; // categoria desconhecida é tratada como ausente
  }
  return v <= arvore.t[i] ? arvore.e[i] : arvore.d[i];
}

function percorrer(arvore, x, modelo) {
  let i = 0;
  while (!arvore.folha[i]) i = passo(arvore, i, x[arvore.f[i]], modelo);
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
// perfil de referência. As contribuições somam prob(pessoa) − v(∅).
// grupos: [{ nome, colunas: [...] }]. Colunas fora do modelo são ignoradas.
//
// Para ser rápido, cada árvore é percorrida uma vez por perfil de referência,
// dividindo o caminho só onde a pessoa e a referência discordam (ver abaixo).
export function explicar(modelo, entrada, fundo, grupos) {
  const colunas = [...modelo.categoricas, ...modelo.numericas];
  const usados = grupos.map(g => ({ ...g, idx: g.colunas.map(c => colunas.indexOf(c)).filter(i => i >= 0) }))
                       .filter(g => g.idx.length);
  const k = usados.length, total = 1 << k;
  const grupoDe = new Int32Array(colunas.length).fill(-1);
  usados.forEach((g, gi) => { for (const i of g.idx) grupoDe[i] = gi; });
  const mascaraArvore = modelo.arvores.map(a => {
    let m = 0;
    for (let n = 0; n < a.f.length; n++) if (!a.folha[n] && grupoDe[a.f[n]] >= 0) m |= 1 << grupoDe[a.f[n]];
    return m;
  });

  const x = vetorizar(modelo, entrada);
  const v = new Float64Array(total);
  const bruto = new Float64Array(total);
  const cache = new Float64Array(total);
  for (const ref of fundo) {
    const b = vetorizar(modelo, ref);
    bruto.fill(modelo.base);
    modelo.arvores.forEach((arvore, t) => {
      const G = mascaraArvore[t];
      // Percorre a árvore uma vez, seguindo a pessoa (x) e a referência (b) ao
      // mesmo tempo. Onde as duas se separam num atributo do grupo g, o caminho
      // se divide: um lado vale quando g está em S, o outro quando não está.
      // Cada folha alcançada vale para os S compatíveis com as escolhas feitas.
      const andar = (i, dentro, fora) => {
        while (!arvore.folha[i]) {
          const f = arvore.f[i], g = grupoDe[f];
          const ib = passo(arvore, i, b[f], modelo);
          if (g < 0 || fora >> g & 1) { i = ib; continue; }
          const ix = passo(arvore, i, x[f], modelo);
          if (ix === ib || dentro >> g & 1) { i = ix; continue; }
          andar(ix, dentro | 1 << g, fora);
          i = ib; fora |= 1 << g;
        }
        const livre = G & ~dentro & ~fora, valor = arvore.v[i];
        let sub = livre;
        while (true) { cache[dentro | sub] = valor; if (sub === 0) break; sub = (sub - 1) & livre; }
      };
      andar(0, 0, 0);
      for (let s = 0; s < total; s++) bruto[s] += cache[s & G];
    });
    for (let s = 0; s < total; s++) {
      const p = 1 / (1 + Math.exp(-bruto[s]));
      v[s] += interpolar(modelo.calibracao.x, modelo.calibracao.y, p);
    }
  }
  for (let s = 0; s < total; s++) v[s] /= fundo.length;

  const fat = [1];
  for (let i = 1; i <= k; i++) fat[i] = fat[i - 1] * i;
  const bits = s => { let c = 0; while (s) { c += s & 1; s >>= 1; } return c; };
  const contribuicoes = usados.map((g, i) => {
    let phi = 0;
    for (let s = 0; s < total; s++) {
      if (s >> i & 1) continue;
      const t = bits(s);
      phi += fat[t] * fat[k - t - 1] / fat[k] * (v[s | 1 << i] - v[s]);
    }
    return { nome: g.nome, valor: phi };
  });
  return { referencia: v[0], prob: v[total - 1], contribuicoes };
}

const GRADUACAO = new Set(["Tecnologia", "Licenciatura", "Bacharelado"]);

export function nivelDoCurso(tipoDeCurso) {
  return GRADUACAO.has(tipoDeCurso) ? "Graduação" : "Técnico";
}

// Limiar de alerta: por nível e, quando se sabe se a renda foi declarada, também por isso.
// Sem essa informação ("Não lembro"), vale o limiar do nível.
export function limiarDoNivel(modelo, tipoDeCurso, rendaNaoDeclarada) {
  const nivel = nivelDoCurso(tipoDeCurso);
  const chave = `${nivel}|${rendaNaoDeclarada}`;
  if (modelo.limiar_renda && (rendaNaoDeclarada === 0 || rendaNaoDeclarada === 1) && chave in modelo.limiar_renda) {
    return modelo.limiar_renda[chave];
  }
  return modelo.limiar[nivel];
}
