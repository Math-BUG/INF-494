# INF-494 - DBSCAN em GPU com CUDA/NVCC

Repositorio do trabalho de GPU sobre adaptacoes experimentais do `DBSCAN`. A entrega usa Google Colab, `CUDA C++`, `%%writefile`, compilacao com `nvcc -O3`, execucao do binario e baseline com `cuML` quando disponivel. Se `cuML` nao estiver disponivel, os notebooks usam `sklearn` como alternativa.

As implementacoes sao didaticas e ainda usam comparacao par-a-par, portanto continuam `O(n^2)`. Por isso os datasets grandes sao tratados por amostragem controlada, e falhas por tempo ou memoria devem ser registradas de forma honesta.

## Protocolo experimental comum

O repositorio agora usa um protocolo comum para evitar comparacoes injustas entre notebooks. O notebook `00_datasets_e_configuracao.ipynb` gera `data/manifest.csv`, e os notebooks de estrategia carregam os mesmos datasets, tamanhos, normalizacao, `eps`, `min_samples`, seed e metricas.

O protocolo padroniza:

- normalizacao `float32` por min-max;
- `estimate_eps(X, min_samples=8, quantile=0.90, sample_size=5000)`;
- `MIN_SAMPLES = 8`;
- `SEED = 42`;
- datasets sinteticos e reais amostrados;
- tamanhos `quick` e `benchmark`;
- metricas `ARI`, `NMI`, porcentagem de ruido, numero de clusters e tempos.

## Modos de execucao

| Modo | Uso | Tamanhos |
| --- | --- | --- |
| `quick` | Depuracao rapida no Colab. | `N_SAMPLES = 4000` |
| `benchmark` | Resultados finais, quando viavel. | `10000`, `25000`, `50000` |

O tamanho `100000` fica como opcional no notebook 00. Nao e recomendado prometer esse tamanho sem testar na GPU disponivel.

## Datasets

Datasets sinteticos controlados:

- `dense_blobs_2d`: gargalo com muitos core points;
- `heterogeneous_blobs_2d`: regioes com densidades diferentes;
- `dense_blobs_noise_2d`: ruido/outliers;
- `moons_2d`: formato nao convexo;
- `blobs_32d`: alta dimensao para quantizacao e leitura de memoria.

Datasets realistas amostrados:

- `real_covtype_sample`: amostra do `fetch_covtype`;
- `real_kddcup99_sample`: amostra do `fetch_kddcup99`, com atributos categoricos convertidos para numeros;
- `real_higgs_sample`: opcional, via `fetch_openml`, desativado por padrao porque pode falhar por rede ou tamanho.

Os arquivos `data/*.bin` e `data/*.npy` sao gerados no Colab e nao sao commitados. O repositorio inclui apenas `data/manifest_exemplo.csv`; o `data/manifest.csv` real deve ser criado executando o notebook 00.

## Notebooks

| Notebook | Funcao |
| --- | --- |
| `00_datasets_e_configuracao.ipynb` | Gera datasets, estima `eps`, salva `.bin`, `.npy`, `data/manifest.csv` e `results/resumo_datasets.csv`. |
| `01_dbscan_nvcc_drop_core_points.ipynb` | Testa drop de core points com `keep_100`, `keep_75`, `keep_50` e `keep_25`. |
| `02_dbscan_nvcc_quantizacao_uint8_uint4.ipynb` | Compara `cuda_cpp_float32`, `cuda_cpp_uint8` e `cuda_cpp_uint4_packed`. |
| `03_benchmark_comparativo.ipynb` | Benchmark consolidado com protocolo comum; registra falhas sem interromper tudo. |
| `04_dbscan_nvcc_multi_eps.ipynb` | Testa multi-EPS experimental com `eps_baixo`, `eps_base` e `eps_alto`. |

## Como reproduzir

1. Abra `00_datasets_e_configuracao.ipynb` no Google Colab.
2. Ative GPU em `Ambiente de execucao > Alterar tipo de ambiente de execucao > GPU`.
3. Rode o notebook 00 para gerar `data/manifest.csv`.
4. Rode os notebooks 01, 02 e 04 individualmente, ou rode `03_benchmark_comparativo.ipynb` para a comparacao consolidada.
5. Confira os CSVs gerados em `results/`.

Todos os notebooks incluem diagnostico:

- `!nvidia-smi`;
- `!nvcc --version`;
- `!which nvprof || echo "nvprof nao encontrado"`.

Quando `nvprof` nao existe, o notebook executa o binario diretamente. O codigo CUDA mede tempo com `cudaEvent`.

## Principais estrategias

| Estrategia | Ideia | Risco |
| --- | --- | --- |
| Drop de core points | Reduzir a quantidade de core points usados em `connect_cores`. | Pode fragmentar clusters, aumentar ruido e reduzir `ARI`/`NMI`. |
| `uint8` | Usar 1 byte por atributo apos normalizacao. | Pode alterar vizinhancas perto de `eps`. |
| `uint4 packed` | Guardar dois atributos por byte. | Pode ser mais rapido, mas tende a perder mais qualidade. |
| Datasets heterogeneos | Testar densidades diferentes e escolha de `eps`. | Um unico `eps` pode nao representar todas as regioes. |
| Multi-EPS | Calcular tres valores de `eps` na mesma execucao CUDA. | Ainda e `O(n^2)` e nao usa indice espacial. |

## Resultados

Resultados finais devem vir do protocolo comum. Se ainda nao houver execucao completa no Colab, use `A preencher apos execucao`.

| Estrategia | Dataset | N | Tempo baseline | Tempo CUDA | Speedup | ARI vs baseline | Observacao |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Drop de core points | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Usar CSV `resultados_drop_core_points_*`. |
| Quantizacao `uint8` | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Comparar contra `cuda_cpp_float32`. |
| Quantizacao `uint4 packed` | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Avaliar perda de qualidade. |
| Multi-EPS | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Comparar contra tres baselines separados. |
| Benchmark consolidado | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Ver `benchmark_comparativo.csv`. |

Um CSV antigo de drop de core points sem metadados de dataset pode existir fora do repositorio local, mas ele nao substitui o benchmark comum porque nao registra `dataset_name`, `n_samples` e `n_features`.

## Limitacoes conhecidas

- As versoes CUDA sao experimentais e `O(n^2)`.
- Nao ha indice espacial.
- `50000` pontos pode falhar dependendo da GPU do Colab e da dimensao.
- Datasets reais sao amostrados para manter custo viavel.
- Nao ha promessa de superar `cuML`; isso deve ser medido.
- Arquivos `.bin` e `.npy` sao gerados localmente e nao ficam no GitHub.
