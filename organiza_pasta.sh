#!/usr/bin/env bash
# Organiza a pasta do trabalho. Não apaga nada: só move.
# Arquivos que já estão no git são movidos com "git mv", para manter o histórico.
# Uso, na raiz do repositório:  bash organiza_pasta.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -d .git ] || { echo "Rode na raiz do repositório (a pasta que tem .git)."; exit 1; }

mkdir -p dados/brutos dados/processados scripts historico/antes_da_reorganizacao referencias

mover() {
  local origem="$1" destino="$2"
  [ -e "$origem" ] || return 0
  if [ -e "$destino" ]; then
    echo "  mantido no lugar (o destino já existe): $origem"
    return 0
  fi
  if git ls-files --error-unmatch "$origem" >/dev/null 2>&1; then
    git mv "$origem" "$destino"
  else
    mv "$origem" "$destino"
  fi
  echo "  $origem -> $destino"
}

echo "Scripts, na ordem em que rodam:"
mover preprocessa_pnp_v2.py   scripts/01_preprocessa_pnp.py
mover junta_edicoes_v2.py     scripts/02_junta_edicoes.py
mover analise_exploratoria.py scripts/03_analise_exploratoria.py
mover modelagem.py            scripts/04_modelagem.py
mover analise_vies.py         scripts/05_analise_vies.py
mover experimentos_melhoria.py scripts/06_experimentos_melhoria.py
mover analise_confianca.py    scripts/07_analise_confianca.py
mover exporta_modelo.py       scripts/08_exporta_modelo.py
mover verifica_site.mjs       scripts/09_verifica_site.mjs

echo "Versões anteriores:"
mover preprocessa_pnp_v1.py historico/preprocessa_pnp_v1.py
mover junta_edicoes_v1.py   historico/junta_edicoes_v1.py
mover saidas_v1             historico/saidas_v1
for f in preprocessa_pnp.py junta_edicoes.py *"("*")"*; do
  [ -e "$f" ] && mover "$f" "historico/antes_da_reorganizacao/$f"
done

echo "Dados (ficam fora do git):"
for f in microdados_*.csv; do mover "$f" "dados/brutos/$f"; done
for f in ef_20*.csv pnp_limpo.csv; do mover "$f" "dados/processados/$f"; done

echo "Documentos de referência:"
for f in *.pdf; do mover "$f" "referencias/$f"; done

if [ -d atualizacao ]; then
  cp -rf atualizacao/. .
  rm -rf atualizacao
  echo "Versões novas dos scripts, do site e do README copiadas por cima."
fi

echo
echo "Ficaram na raiz (confira um a um e mova ou apague você mesma):"
ls -A | grep -v -x -E '\.git|\.gitignore|\.venv|README\.md|requirements\.txt|organiza_pasta\.sh|dados|scripts|historico|referencias|docs|figuras|resultados|coleta' || echo "  nada"
