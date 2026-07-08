#!/bin/bash
set -euo pipefail

module --force purge
module load GCCcore/12.2.0
module load CUDA/12.6.0

export PYTHONNOUSERSITE=1

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_cluster.txt

python - <<'PY'
import site
import sys

print("python:", sys.executable)
print("user site enabled:", site.ENABLE_USER_SITE)
try:
    import torch  # noqa: F401
except Exception:
    print("torch ausente na venv: ok")
else:
    print("AVISO: torch esta instalado nesta venv, mas este benchmark nao usa torch.")
    print("Para instalar cuML sem conflito, recrie a venv limpa antes.")
PY

if [ "${DBSCAN_INSTALL_CUML:-0}" = "1" ]; then
  python -m pip install -r requirements_cuml_cluster.txt
fi

python -c "import numpy, pandas, sklearn, matplotlib; print('deps basicas ok')"

if [ "${DBSCAN_INSTALL_CUML:-0}" = "1" ]; then
  python -c "import cupy; from cuml.cluster import DBSCAN; print('cuml ok')"
fi
