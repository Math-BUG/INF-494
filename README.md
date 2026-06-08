# INF-494 - DBSCAN em GPU com CUDA/NVCC

Este repositorio contem os quatro notebooks finais do trabalho de GPU sobre adaptacoes experimentais do `DBSCAN` para execucao em GPU. O foco e estudar gargalos do algoritmo em datasets densos e heterogeneos, comparar com `cuML` e testar aproximacoes que reduzem custo de memoria ou custo de conexao entre core points.

Os notebooks seguem o estilo usado na disciplina: Google Colab, RAPIDS/`cuML` como baseline quando disponivel, alternativa com `sklearn`, codigo `CUDA C++` escrito com `%%writefile`, compilacao com `nvcc -O3` e execucao do binario. Quando `nvprof` estiver disponivel, ele e usado para perfilamento; quando nao estiver, o binario e executado diretamente, pois o proprio codigo CUDA mede tempo com `cudaEvent`.

## Relacao com o pedido do professor

Pelas fontes locais do trabalho, a orientacao principal foi explorar por que o `DBSCAN` fica caro em datasets densos: muitos pontos viram core points e a etapa de aglomeracao/conexao entre esses pontos passa a comparar muitas vizinhancas sobrepostas. O trabalho tambem pede comparacao com `cuML`, uso da versao de referencia citada em aula como inspiracao metodologica, drop de core points, quantizacao em 8 e 4 bits, datasets com densidade variavel e uma ideia multi-EPS.

O Colab citado na aula nao e copiado aqui. Os notebooks deste repositorio sao autocontidos e implementam as versoes finais diretamente.

## Notebooks publicados

| Notebook | Papel no trabalho |
| --- | --- |
| `01_dbscan_nvcc_drop_core_points.ipynb` | Testa `DBSCAN` em `CUDA C++` com reducao deterministica de core points: 100%, 75%, 50% e 25%. |
| `02_dbscan_nvcc_quantizacao_uint8_uint4.ipynb` | Compara `cuda_cpp_float32`, `cuda_cpp_uint8` e `cuda_cpp_uint4_packed` no mesmo fluxo experimental. |
| `03_dbscan_nvcc_datasets_reais_heterogeneos.ipynb` | Organiza benchmarks com datasets sinteticos controlados e datasets reais complementares do `sklearn`. |
| `04_dbscan_nvcc_multi_eps.ipynb` | Implementa uma versao experimental multi-EPS, calculando tres valores de `eps` na mesma execucao CUDA. |

## Como executar no Colab

1. Abra o notebook desejado no Google Colab.
2. Ative GPU em `Ambiente de execucao > Alterar tipo de ambiente de execucao > GPU`.
3. Execute a celula de diagnostico:
   - `!nvidia-smi`
   - `!nvcc --version`
   - `!which nvprof || echo "nvprof nao encontrado"`
4. Execute as celulas em ordem.
5. Confira se `cuML` foi importado. Se `cuML` nao estiver disponivel, o notebook usa `sklearn` como baseline CPU.
6. Compile o codigo CUDA com `nvcc -O3`.
7. Execute o binario. Se `nvprof` existir, o notebook usa `nvprof`; caso contrario, executa diretamente.
8. As tabelas finais sao salvas em `results/`.

Os notebooks devem ser executados no Google Colab com GPU ativada para preencher os resultados finais. Este repositorio nao inventa numeros de desempenho.

## Principais estrategias implementadas

| Estrategia | Ideia | Risco/limitacao |
| --- | --- | --- |
| Drop de core points | Mantem apenas uma fracao deterministica dos core points para reduzir o custo de `connect_cores`. | Pode fragmentar clusters, aumentar ruido e reduzir `ARI`/`NMI`. |
| Quantizacao `uint8` | Normaliza dados para `[0, 1]` e usa 1 byte por atributo. | Aproxima distancias e pode alterar core points. |
| Quantizacao `uint4 packed` | Usa valores de 0 a 15 e empacota dois atributos por byte. | Maior perda de precisao; qualidade pode cair. |
| Datasets densos/heterogeneos | Testa cenarios onde muitos pontos viram core points e onde densidades diferentes dificultam escolher `eps`. | Datasets sinteticos ajudam a isolar efeitos, mas nao substituem dados reais. |
| Multi-EPS experimental | Calcula `eps_baixo`, `eps_base` e `eps_alto` em uma unica execucao CUDA para reaproveitar leituras/calculos. | Ainda e `O(n^2)`, nao usa indice espacial e ainda nao combina com drop de core points. |

## Resultados esperados

Espera-se que o drop de core points reduza principalmente o tempo de `connect_cores`, com perda gradual de qualidade quando o `keep_ratio` diminui. A quantizacao pode reduzir leitura de memoria, mas tambem pode alterar vizinhancas por arredondamento. A versao multi-EPS deve ser avaliada como demonstracao de reaproveitamento de distancia para varios `eps`, nao como implementacao industrial otimizada.

## Limitacoes conhecidas

- As implementacoes CUDA sao didaticas e usam comparacao par-a-par, portanto continuam com custo `O(n^2)`.
- Nao ha indice espacial como grid, kd-tree ou ball-tree.
- `nvprof` pode nao estar disponivel em algumas imagens recentes do Colab; por isso existe alternativa para execucao direta.
- Os datasets reais pequenos do `sklearn` ajudam na validacao, mas nao sao bons benchmarks de ocupacao de GPU.
- Resultados finais dependem da GPU do Colab e devem ser preenchidos apos execucao real.

## Tabela-resumo dos resultados

| Estrategia | Dataset | Tempo baseline | Tempo CUDA | Speedup | ARI vs baseline | Observacao |
| ---------- | ------- | -------------: | ---------: | ------: | --------------: | ---------- |
| Drop de core points | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Comparar 100%, 75%, 50% e 25%. |
| Quantizacao `uint8` | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Comparar contra `cuda_cpp_float32`. |
| Quantizacao `uint4 packed` | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Avaliar perda de qualidade. |
| Benchmarks de datasets | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Usar datasets sinteticos e reais complementares. |
| Multi-EPS | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | A preencher apos execucao | Comparar contra tres baselines separados. |
