# Benchmark DBSCAN CUDA Multi-parametro

Este projeto executa e compara implementacoes CUDA de DBSCAN para multiplos
valores de `eps` e `minPts` em uma unica rodada. O objetivo e medir o custo de
varrer configuracoes de DBSCAN em GPU e comparar esses tempos com chamadas
equivalentes do `cuML.DBSCAN`.

O benchmark avalia tres modos principais:

- `multi_eps`: uma chamada CUDA avalia varios valores de `eps` mantendo um
  `minPts` de referencia.
- `multi_minpts`: uma chamada CUDA avalia varios valores de `minPts` mantendo
  um `eps` de referencia.
- `multi_both`: uma chamada CUDA avalia a grade completa de `eps x minPts`.

Para a comparacao com cuML, o runner mede tempos equivalentes:

- cuML para 1 configuracao de referencia.
- cuML para os 3 valores de `eps` usados pelo `multi_eps`.
- cuML para os 4 valores de `minPts` usados pelo `multi_minpts`.
- cuML para as 12 combinacoes da grade `3 x 4` usada pelo `multi_both`.

## Estrutura

Arquivos principais:

- `run_dbscan_cluster.py`: runner principal do benchmark.
- `dbscan_multi_baixo_nivel.cu`: codigo CUDA com os modos multi-parametro.
- `datasets_sinteticos.py`: geracao dos datasets sinteticos e sugestao de
  parametros DBSCAN.
- `requirements_cluster.txt`: dependencias Python basicas.
- `requirements_cuml_cluster.txt`: dependencias opcionais para cuML/CuPy.
- `setup_venv_cluster.sh`: script auxiliar para criar a venv.

Os arquivos de submissao para agendadores de jobs nao sao necessarios para
entender ou executar o runner Python. Em ambientes com agendador, use os mesmos
comandos Python descritos aqui dentro do script de submissao apropriado.

## Requisitos

Requisitos de sistema:

- Linux x86_64.
- Python 3.10 ou compativel com RAPIDS/cuML instalado.
- NVIDIA driver funcional.
- CUDA Toolkit com `nvcc`.
- GPU NVIDIA compativel com a arquitetura definida em `--cuda-arch`.

Dependencias Python basicas:

```bash
python -m pip install -r requirements_cluster.txt
```

Dependencias opcionais para comparar com cuML:

```bash
python -m pip install -r requirements_cuml_cluster.txt
```

O arquivo `requirements_cuml_cluster.txt` instala:

- `cupy-cuda12x`
- `cuml-cu12`

Use uma venv limpa para evitar conflitos com outras bibliotecas CUDA, como
instalacoes de PyTorch que fixam versoes diferentes dos pacotes `nvidia-*`.

## Preparacao da Venv

Criacao da venv apenas com as dependencias basicas:

```bash
bash setup_venv_cluster.sh
source .venv/bin/activate
```

Criacao da venv com cuML/CuPy:

```bash
DBSCAN_INSTALL_CUML=1 bash setup_venv_cluster.sh
source .venv/bin/activate
```

Validacao rapida:

```bash
python -c "import numpy, pandas, sklearn; print('deps basicas ok')"
python -c "import cupy; from cuml.cluster import DBSCAN; print('cuml/cupy ok')"
```

Se cuML nao estiver disponivel, o benchmark ainda pode ser executado com
`--skip-cuml`. Nesse caso, as colunas de comparacao com cuML ficam vazias.

## Execucao Rapida

Exemplo pequeno, com um dataset, um tamanho e uma repeticao:

```bash
python run_dbscan_cluster.py \
  --datasets dense_blobs_2d \
  --n-points 4000 \
  --repeticoes 1 \
  --output-dir outputs/teste_rapido \
  --work-dir outputs/teste_rapido/work \
  --cuda-arch 75
```

Sem cuML:

```bash
python run_dbscan_cluster.py \
  --datasets dense_blobs_2d \
  --n-points 4000 \
  --repeticoes 1 \
  --skip-cuml \
  --output-dir outputs/teste_rapido_sem_cuml \
  --work-dir outputs/teste_rapido_sem_cuml/work
```

## Execucao Completa

Exemplo com varios datasets e tamanhos:

```bash
python run_dbscan_cluster.py \
  --datasets heterogeneous_blobs_2d,dense_blobs_2d,spiral_2d,moons_10d,rings_10d \
  --n-points 4000,16000,64000,128000 \
  --repeticoes 3 \
  --output-dir outputs/rodada_principal \
  --work-dir outputs/rodada_principal/work \
  --cuda-arch 75
```

O runner compila automaticamente `dbscan_multi_baixo_nivel.cu` com `nvcc` antes
da primeira execucao. Para reutilizar um binario ja compilado, use
`--no-compile`.

## Execucao por Shards

Para dividir a grade de experimentos em varias partes independentes, use
`--num-shards` e `--shard-index`.

Exemplo com 4 shards:

```bash
python run_dbscan_cluster.py \
  --num-shards 4 \
  --shard-index 0 \
  --output-dir outputs/rodada_shards \
  --work-dir outputs/rodada_shards/work_0
```

```bash
python run_dbscan_cluster.py \
  --num-shards 4 \
  --shard-index 1 \
  --output-dir outputs/rodada_shards \
  --work-dir outputs/rodada_shards/work_1
```

```bash
python run_dbscan_cluster.py \
  --num-shards 4 \
  --shard-index 2 \
  --output-dir outputs/rodada_shards \
  --work-dir outputs/rodada_shards/work_2
```

```bash
python run_dbscan_cluster.py \
  --num-shards 4 \
  --shard-index 3 \
  --output-dir outputs/rodada_shards \
  --work-dir outputs/rodada_shards/work_3
```

Cada shard gera seu proprio checkpoint:

```text
resultados_brutos_cuml_justo_shard00_of_04.csv
resultados_brutos_cuml_justo_shard01_of_04.csv
resultados_brutos_cuml_justo_shard02_of_04.csv
resultados_brutos_cuml_justo_shard03_of_04.csv
```

Depois que os shards terminarem, faca o merge:

```bash
python run_dbscan_cluster.py \
  --output-dir outputs/rodada_shards \
  --num-shards 4 \
  --merge-only
```

O merge cria:

```text
resultados_brutos_cuml_justo.csv
resultados_agregados_cuml_justo.csv
```

## Checkpoints e Retomada

O runner salva checkpoint apos cada combinacao concluida. A chave de retomada e:

```text
dataset + N + repeticao
```

Se uma execucao for interrompida, rode novamente com o mesmo `--output-dir`,
mesmo `--num-shards` e mesmo `--shard-index`. O runner pula as combinacoes ja
presentes no CSV de checkpoint.

Para forcar uma nova execucao sem retomar checkpoint:

```bash
python run_dbscan_cluster.py \
  --output-dir outputs/nova_rodada \
  --no-resume
```

## Datasets

Os datasets sinteticos sao definidos em `datasets_sinteticos.py`.

Familias disponiveis no benchmark:

- `heterogeneous_blobs`
- `dense_blobs`
- `spiral`
- `moons`
- `rings`

Exemplos de nomes aceitos:

```text
heterogeneous_blobs_2d
heterogeneous_blobs_12d
dense_blobs_16d
spiral_14d
moons_10d
rings_16d
```

O runner usa `datasets_sinteticos.py` para gerar a matriz `X` e sugerir os
valores base de `eps` e `minPts`. A partir desses valores, ele monta:

- 3 valores de `eps`.
- 4 valores de `minPts`.
- grade completa com 12 configuracoes.

## Saidas

O CSV bruto tem uma linha por execucao:

```text
dataset, D, N, repeticao
```

Principais colunas de tempo:

```text
tempo_multi_eps_ms
tempo_multi_minpts_ms
tempo_multi_both_ms
tempo_cuml_1_chamada_ms
tempo_cuml_multi_eps_3_chamadas_ms
tempo_cuml_multi_minpts_4_chamadas_ms
tempo_cuml_multi_both_12_chamadas_ms
```

Principais colunas de speedup:

```text
speedup_multi_eps_vs_cuml_3_eps
speedup_multi_minpts_vs_cuml_4_minpts
speedup_multi_both_vs_cuml_12_combinacoes
```

Interpretacao:

```text
speedup = tempo_cuml_equivalente / tempo_cuda_multi
```

Valor maior que `1` indica que o modo CUDA multi-parametro foi mais rapido que
as chamadas equivalentes do cuML.

O CSV agregado resume as repeticoes por:

```text
dataset + D + N
```

e calcula media e desvio padrao das metricas numericas.

## Metodologia de Medicao

Para os kernels CUDA proprios, os tempos reportados vem do binario CUDA e sao
baseados em eventos CUDA. O tempo registrado foca a execucao em GPU, nao o custo
total de preparar dados, gravar arquivos temporarios ou salvar CSVs.

Para o cuML, o runner usa CuPy para transferir a matriz para GPU uma vez:

```python
x_matrix_gpu = cupy.asarray(x_matrix)
```

Depois mede cada chamada `fit` com eventos CUDA do CuPy. Antes da medicao real,
uma chamada de warm-up e executada e descartada. Assim, a comparacao principal
fica centrada no tempo de execucao em GPU com dados ja residentes no dispositivo.

Essa metodologia e adequada para comparar o custo computacional dos algoritmos.
Ela nao representa o tempo total percebido pelo usuario em um pipeline completo
com geracao de dados, I/O, transferencia CPU-GPU, escrita de labels e salvamento
dos resultados.

## Profiling com nvprof

O runner pode salvar a saida do `nvprof` para cada chamada CUDA:

```bash
python run_dbscan_cluster.py \
  --datasets dense_blobs_2d \
  --n-points 16000 \
  --repeticoes 1 \
  --profile-nvprof \
  --output-dir outputs/perfil_nvprof
```

Por padrao, os perfis ficam em:

```text
outputs/perfil_nvprof/nvprof/
```

Tambem e possivel informar outro diretorio:

```bash
python run_dbscan_cluster.py \
  --profile-nvprof \
  --nvprof-dir outputs/nvprof_custom
```

Atencao: profiling adiciona overhead. Use `nvprof` em rodadas pequenas para
diagnostico, nao em toda a grade principal.

## Parametros do Runner

Parametros mais usados:

```text
--datasets              lista de datasets separada por virgula
--n-points              lista de tamanhos N separada por virgula
--repeticoes            numero de repeticoes por dataset/N
--output-dir            diretorio dos CSVs de saida e checkpoints
--work-dir              diretorio temporario para binarios e arquivos .bin
--cuda-arch             arquitetura CUDA usada no nvcc, por exemplo 75, 80, 86
--timeout-kernel-s      timeout por chamada ao binario CUDA
--skip-cuml             pula comparacao com cuML
--profile-nvprof        salva saida do nvprof
--print-nvprof          imprime stderr do nvprof no log
--nvprof-dir            diretorio de saida dos arquivos nvprof
--no-resume             ignora checkpoints anteriores
--no-compile            nao recompila o codigo CUDA
--num-shards            numero total de shards
--shard-index           indice do shard atual
--merge-only            apenas junta CSVs de shards ja executados
```

## Reprodutibilidade

O runner usa uma seed base configuravel:

```bash
python run_dbscan_cluster.py --seed 42
```

Cada combinacao de dataset, tamanho e repeticao deriva seus dados a partir dessa
seed, mantendo a rodada reproduzivel quando os mesmos parametros sao usados.

Para comparar resultados entre maquinas ou execucoes, mantenha constantes:

- versao do codigo CUDA;
- versao de CUDA Toolkit;
- versao do driver NVIDIA;
- versao de cuML/CuPy;
- `--cuda-arch`;
- datasets, tamanhos e repeticoes;
- numero de shards e mapeamento dos shards.

## Observacoes de Performance

DBSCAN exato baseado em comparacoes de distancia pode ter custo quadratico em
relacao a `N`. Portanto, tamanhos grandes como `N=1_000_000` ou maiores podem
demorar muito, especialmente quando combinados com muitos datasets, repeticoes e
comparacoes cuML equivalentes.

Para validar a configuracao, comece com uma grade pequena:

```bash
python run_dbscan_cluster.py \
  --datasets dense_blobs_2d \
  --n-points 4000,16000 \
  --repeticoes 1 \
  --output-dir outputs/validacao
```

Depois aumente `N`, numero de datasets e repeticoes progressivamente.

## Troubleshooting

Erro ao importar cuML:

```bash
python run_dbscan_cluster.py --skip-cuml
```

Erro de compilacao CUDA:

- verifique se `nvcc` esta no `PATH`;
- confirme a arquitetura passada em `--cuda-arch`;
- teste com um `N` pequeno antes da rodada completa.

Timeout do kernel:

```bash
python run_dbscan_cluster.py --timeout-kernel-s 14400
```

Rodada interrompida:

- rode novamente com o mesmo `--output-dir`;
- nao use `--no-resume`;
- mantenha o mesmo `--num-shards` e `--shard-index`.

Resultados parciais por shard:

```bash
python run_dbscan_cluster.py \
  --output-dir outputs/rodada_shards \
  --num-shards 4 \
  --merge-only
```
