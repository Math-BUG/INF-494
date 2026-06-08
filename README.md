# INF-494 - DBSCAN em GPU com CUDA/NVCC

Repositorio dos quatro notebooks finais do trabalho de GPU sobre adaptacoes do DBSCAN.

Os notebooks seguem o estilo usado na disciplina: Colab, RAPIDS/cuML como baseline quando disponivel, CUDA C++ em arquivos `.cu`, compilacao com `nvcc -O3` e execucao/perfilamento com `nvprof`.

## Notebooks publicados

1. `01_dbscan_nvcc_drop_core_points.ipynb`
   - DBSCAN em CUDA C++ com reducao deterministica de core points.
   - Compara `keep_ratio` de 100%, 75%, 50% e 25%.

2. `02_dbscan_nvcc_quantizacao_uint8_uint4.ipynb`
   - DBSCAN em CUDA C++ usando dados quantizados.
   - Testa `float32`, `uint8` e `uint4 packed`.

3. `03_dbscan_nvcc_datasets_reais_heterogeneos.ipynb`
   - Testes com datasets reais do `sklearn`.
   - Foco em heterogeneidade e variacao de densidade.

4. `04_dbscan_nvcc_multi_eps.ipynb`
   - Versao experimental multi-EPS inspirada na ideia de Multi-K.
   - Executa multiplos valores de `eps` em uma mesma estrutura CUDA.

## Como executar

Abra cada notebook no Google Colab com GPU ativada. Em geral, a ordem de execucao e:

1. verificar GPU e bibliotecas;
2. gerar/carregar o dataset;
3. rodar baseline com cuML, ou sklearn caso cuML nao esteja disponivel;
4. gravar o codigo CUDA com `%%writefile`;
5. compilar com `!nvcc ... -O3`;
6. executar com `!nvprof ./...`;
7. comparar tempo, speedup e metricas de qualidade.

Os arquivos de apoio, fontes anexadas e notebooks de rascunho ficam fora da publicacao para manter o repositorio limpo.
