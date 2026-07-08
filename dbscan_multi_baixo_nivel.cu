// =============================================================================
// Implementacao de DBSCAN em CUDA (GPU), com varias "variantes" de kernel:
//   - "seq"         : um unico par (eps, minPts)
//   - "multi_eps"   : varios valores de eps, minPts fixo
//   - "multi_minpts": eps fixo, varios valores de minPts
//   - "multi_both"  : varios eps x varios minPts ao mesmo tempo (grade combinada)
//
// Ideia geral do algoritmo (DBSCAN classico, mas paralelizado):
//   1) Para cada ponto, conta quantos vizinhos existem dentro do raio eps
//      (kernels count_neighbors_*).
//   2) Marca como "core" (nucleo) todo ponto cuja contagem >= minPts
//      (kernels mark_core_*).
//   3) Conecta pontos "core" que estao a distancia <= eps um do outro,
//      usando Union-Find (Disjoint Set Union) para formar os clusters
//      (kernels connect_core_*, com find/union via path splitting).
//   4) "Achata" a arvore de Union-Find para que cada ponto aponte
//      diretamente para a raiz do seu cluster (flatten_parent_kernel).
//   5) Atribui pontos "borda" (nao-core, mas vizinhos de um core) ao
//      cluster do core mais proximo (kernels assign_border_*).
//
// As variantes "multi_*" fazem tudo isso para varias combinacoes de
// parametros de uma vez so, para nao precisar rodar o kernel do zero
// para cada combinacao de eps/minPts (evita reler os dados da GPU
// repetidas vezes).
// =============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

// Macro de checagem de erro: toda chamada CUDA relevante passa por aqui.
// Se der erro, imprime arquivo/linha e a mensagem, e encerra o programa.
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

// Limites usados para decidir se certos vetores (coordenadas do ponto,
// valores de eps) cabem em registradores da thread (mais rapido) ou
// precisam ficar em memoria global/shared (mais lento, mas sem limite de tamanho).
#define MAX_D_REG 32       // dimensao maxima para guardar o ponto "xi" em registrador
#define MAX_PARAM_REG 32   // quantidade maxima de parametros (eps ou minPts) em registrador
#define DEFAULT_MAX_GRAPH_MBYTES 2048.0


// -----------------------------------------------------------------------
// UNION-FIND (Disjoint Set Union) EM GPU
// -----------------------------------------------------------------------

__device__ int find_root_offset(int* parent, int base, int i) {
    // Path splitting: cada thread encurta o caminho apontando o no atual para o avo.
    // Como union_roots_offset sempre liga a raiz maior na menor, a arvore permanece aciclica
    // e a busca converge sem guard fixo fragil.
    //
    // 'base' e um deslocamento no vetor 'parent' -- usado quando ha varios
    // Union-Finds independentes armazenados lado a lado no mesmo array
    // (um por combinacao de eps/minPts).
    int x = i;
    while (true) {
        int p = parent[base + x];       // pai atual de x
        int gp = parent[base + p];      // "avo" de x (pai do pai)
        if (p == gp) return p;          // p e a raiz (aponta para si mesmo) -> retorna
        atomicCAS(&parent[base + x], p, gp); // tenta encurtar o caminho: x -> avo
        x = p;                          // continua subindo a arvore
    }
}

__device__ void union_roots_offset(int* parent, int base, int a, int b) {
    // Thread-safe com atomicCAS e politica deterministica: maior raiz aponta para menor raiz.
    // Isso evita ciclos e race conditions entre threads diferentes tentando unir
    // os mesmos dois conjuntos ao mesmo tempo.
    while (true) {
        int ra = find_root_offset(parent, base, a);
        int rb = find_root_offset(parent, base, b);
        if (ra == rb) return; // ja estao no mesmo conjunto, nada a fazer
        int hi = (ra > rb) ? ra : rb; // raiz maior
        int lo = (ra > rb) ? rb : ra; // raiz menor
        // tenta trocar parent[hi] de "hi" (aponta pra si mesmo) para "lo".
        // Se outra thread mudou 'hi' entretanto, o CAS falha e o loop tenta de novo.
        int old = atomicCAS(&parent[base + hi], hi, lo);
        if (old == hi) return;
    }
}


// -----------------------------------------------------------------------
// FUNCOES DE DISTANCIA (distancia euclidiana ao quadrado, com "early exit")
// -----------------------------------------------------------------------

__device__ float dist_sq_cutoff(const float* X, int i, int j, int d, float cutoff) {
    // Distancia euclidiana^2 entre os pontos i e j, direto da memoria global.
    // Para de somar assim que ultrapassa o 'cutoff' (eps^2) -- economiza
    // trabalho quando os pontos estao claramente longe demais.
    float s = 0.0f;
    for (int f = 0; f < d; f++) {
        float diff = X[(long long)i * d + f] - X[(long long)j * d + f];
        s += diff * diff;
        if (s > cutoff) break;
    }
    return s;
}

__device__ float dist_sq_tile_cutoff(const float* xi_reg, const float* X, const float* tile_X, int i, int local_j, int d, float cutoff, bool xi_em_reg) {
    // Versao "tiled": o ponto j vem da memoria compartilhada (tile_X), que e
    // muito mais rapida que a memoria global. O ponto i pode vir de registrador
    // (xi_reg, se coube) ou de memoria global (fallback para dimensoes grandes).
    float s = 0.0f;
    long long base_i = (long long)i * d;
    long long base_j = (long long)local_j * d;
    for (int f = 0; f < d; f++) {
        float xi = xi_em_reg ? xi_reg[f] : X[base_i + f];
        float diff = xi - tile_X[base_j + f];
        s += diff * diff;
        if (s > cutoff) break;
    }
    return s;
}


// -----------------------------------------------------------------------
// HELPERS PARA MULTIPLOS VALORES DE EPS / MINPTS AO MESMO TEMPO
// -----------------------------------------------------------------------

__device__ int first_core_eps_index(const int* core, int n, int e_count, int p) {
    // Core por eps e monotonico: se e core em eps menor, tambem e core nos maiores.
    // Entao basta achar o PRIMEIRO indice de eps (o menor) em que o ponto p
    // ja e core -- a partir dali, ele e core para todo eps maior tambem.
    for (int e = 0; e < e_count; e++) {
        if (core[e * n + p]) return e;
    }
    return e_count; // nunca e core em nenhum eps testado
}

__device__ int core_limit_minpts_index(const int* core, int n, int m_count, int p) {
    // minPts esta ordenado crescente. Quando deixa de ser core, nao volta a ser core.
    // Retorna o MAIOR indice de minPts em que o ponto p ainda e core
    // (quanto maior minPts, mais dificil ser core, entao a lista de "core" so pode diminuir).
    int limit = -1;
    for (int m = 0; m < m_count; m++) {
        if (core[m * n + p]) limit = m;
        else break; // assim que falha, para (monotonico)
    }
    return limit;
}

__device__ unsigned long long pack_core_limits_eps_minpts(const int* core, int n, int e_count, int m_count, int p) {
    // Em cada nibble, guarda maior indice de minPts ainda core para um eps.
    // 0 = sem core; k+1 = core ate minPts index k. Caminho comum do notebook cabe em 64 bits.
    //
    // Isso comprime, para um ponto p, toda a informacao "ate que minPts ele
    // ainda e core, para cada eps" em um unico inteiro de 64 bits (16 nibbles
    // de 4 bits, um por eps, cobrindo ate 15 valores de minPts por eps).
    // Serve para nao precisar reler o array 'core' inteiro depois.
    unsigned long long packed = 0ULL;
    int max_e = e_count < 16 ? e_count : 16;
    int max_m = m_count < 15 ? m_count : 15;
    for (int e = 0; e < max_e; e++) {
        int limit_plus_one = 0;
        int combo_base = e * m_count;
        for (int m = 0; m < max_m; m++) {
            if (core[(combo_base + m) * n + p]) limit_plus_one = m + 1;
            else break;
        }
        packed |= ((unsigned long long)(limit_plus_one & 15)) << (4 * e);
    }
    return packed;
}

__device__ int unpack_core_limit(unsigned long long packed, int e) {
    // Le de volta o nibble referente ao eps 'e' e subtrai 1
    // (porque 0 significava "sem core", entao o valor real e limit_plus_one - 1).
    return (int)((packed >> (4 * e)) & 15ULL) - 1;
}

__device__ int first_eps_by_distance_repeated_cutoff(const float* xi_reg, const float* X, const float* tile_X, const float* eps2_values, int i, int local_j, int d, int first_e, int e_count, bool xi_em_reg) {
    // Para alta dimensao: usa cutoff proprio de cada eps. Isso pode repetir trabalho,
    // mas evita que um eps grande obrigue todos os eps menores a lerem todas as dimensoes.
    //
    // Em vez de calcular a distancia UMA VEZ com o maior eps (o que forcaria
    // ler todas as 'd' dimensoes mesmo quando o eps pequeno jÃ¡ bastaria para
    // decidir "esta fora"), recalcula a distancia para cada eps com seu
    // proprio cutoff -- em alta dimensao, o "early exit" da funcao de
    // distancia compensa o recalculo.
    for (int e = first_e; e < e_count; e++) {
        float eps2 = eps2_values[e];
        float dist = dist_sq_tile_cutoff(xi_reg, X, tile_X, i, local_j, d, eps2, xi_em_reg);
        if (dist <= eps2) return e; // primeiro eps (o menor a partir de first_e) que conecta
    }
    return e_count; // nao conecta em nenhum eps
}


// =============================================================================
// KERNELS DO CASO SIMPLES ("seq"): um unico eps e um unico minPts
// =============================================================================

__global__ void init_parent_kernel(int* parent, int groups, int n) {
    // Inicializa o Union-Find: cada ponto comeca sendo seu proprio pai (raiz).
    // 'groups' permite inicializar varios Union-Finds de uma vez (um por
    // combinacao de parametros), todos concatenados no mesmo array.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * n;
    if (idx < total) parent[idx] = idx % n;
}

__global__ void count_neighbors_single_eps_kernel(const float* X, int* counts, int n, int d, float eps2) {
    // Para cada ponto i, conta quantos pontos j estao a distancia <= eps.
    // Usa "tiling": carrega um bloco (tile) de pontos por vez na memoria
    // compartilhada (shared memory), que e muito mais rapida que a global,
    // e cada thread do bloco compara seu ponto 'i' contra todos os pontos do tile.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG; // se a dimensao couber, guarda o ponto em registrador
    float xi[MAX_D_REG];
    if (xi_em_reg) {
        for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    }

    int c = 0; // contador de vizinhos dentro do eps
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        // cada thread do bloco ajuda a copiar um pedaco do tile para shared memory
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) {
            tile_X[k] = X[(long long)tile_start * d + k];
        }
        __syncthreads(); // espera todo mundo terminar de copiar antes de usar o tile

        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, eps2, xi_em_reg);
                if (dist <= eps2) c++;
            }
        }
        __syncthreads(); // espera todo mundo terminar de ler antes de sobrescrever o tile
    }
    if (valido) counts[i] = c;
}

__global__ void mark_core_single_kernel(const int* counts, int* core, int n, int min_pts) {
    // Marca 1 (core) se o numero de vizinhos >= minPts, senao 0.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) core[i] = (counts[i] >= min_pts) ? 1 : 0;
}

__global__ void connect_cores_single_kernel(const float* X, const int* core, int* parent, int n, int d, float eps2) {
    // Une (via Union-Find) todo par de pontos "core" que estao dentro de eps
    // um do outro. So threads cujo ponto 'i' e core participam ativamente.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool ativo = (i < n && core[i]);
    bool xi_em_reg = ativo && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) {
        for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    }

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        // se NENHUMA thread do bloco tem trabalho a fazer, sai do loop mais cedo
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) {
            tile_X[k] = X[(long long)tile_start * d + k];
        }
        __syncthreads();

        if (ativo) {
            int first_local = 0;
            if (tile_start <= i) {
                // evita comparar o par (i,j) duas vezes (uma como i->j e outra
                // como j->i): so olha para j > i dentro do tile
                first_local = i - tile_start + 1; // preserva j > i e evita pares duplicados.
                if (first_local < 0) first_local = 0;
                if (first_local > tile_count) first_local = tile_count;
            }
            for (int local_j = first_local; local_j < tile_count; local_j++) {
                int j = tile_start + local_j;
                if (!core[j]) continue; // so une core com core
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, eps2, xi_em_reg);
                if (dist <= eps2) union_roots_offset(parent, 0, i, j);
            }
        }
        __syncthreads();
    }
}


__global__ void flatten_parent_kernel(int* parent, int groups, int n) {
    // Depois de todas as unioes, "achata" a arvore: cada ponto passa a
    // apontar direto para a raiz do seu cluster (em vez de precisar
    // percorrer varios niveis toda vez que alguem consultar 'parent').
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = groups * n;
    if (idx >= total) return;
    int base = (idx / n) * n; // deslocamento do Union-Find deste 'grupo' (combinacao de parametros)
    int i = idx % n;
    parent[base + i] = find_root_offset(parent, base, i);
}

__global__ void assign_borders_single_kernel(const float* X, const int* core, const int* parent, int* labels, int n, int d, float eps2) {
    // Atribui o rotulo final de cada ponto:
    //  - se o ponto e core, seu rotulo e o proprio cluster (parent[i])
    //  - senao, procura um vizinho core dentro de eps e "herda" o cluster dele (ponto de borda)
    //  - se nao achar nenhum core por perto, fica com rotulo -1 (ruido)
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool done = false;
    int assigned = -1;
    if (valido && core[i]) { assigned = parent[i]; done = true; }

    bool xi_em_reg = valido && !done && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int ativo = (valido && !done); // so continua procurando quem ainda nao tem rotulo
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        __syncthreads();

        if (ativo) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                int j = tile_start + local_j;
                if (!core[j]) continue; // so pode "herdar" cluster de um ponto core
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, eps2, xi_em_reg);
                if (dist <= eps2) { assigned = parent[j]; done = true; break; }
            }
        }
        __syncthreads();
    }
    if (valido) labels[i] = assigned;
}


// =============================================================================
// KERNELS DO CASO "multi_eps": varios valores de eps, minPts fixo
// =============================================================================

__global__ void count_neighbors_multi_eps3_kernel(const float* X, const float* eps2_values, int* counts, int n, int d) {
    // Versao especializada para exatamente 3 valores de eps (caso comum no notebook).
    // Calcula a distancia UMA VEZ (com cutoff no maior eps, e2) e classifica
    // o resultado nos 3 contadores de uma vez, aproveitando que eps0<=eps1<=eps2.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float e0 = eps2_values[0], e1 = eps2_values[1], e2 = eps2_values[2];
    int c0 = 0, c1 = 0, c2 = 0;
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        __syncthreads();
        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                // usa o MAIOR eps (e2) como cutoff da distancia -- so calcula uma vez
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, e2, xi_em_reg);
                // como eps0 <= eps1 <= eps2, se esta dentro do menor, esta dentro de todos
                if (dist <= e0) { c0++; c1++; c2++; }
                else if (dist <= e1) { c1++; c2++; }
                else if (dist <= e2) { c2++; }
            }
        }
        __syncthreads();
    }
    if (valido) { counts[i] = c0; counts[n + i] = c1; counts[2 * n + i] = c2; }
}

__global__ void count_neighbors_multi_eps3_hidim_kernel(const float* X, const float* eps2_values, int* counts, int n, int d) {
    // Variante "hidim" (alta dimensao) do kernel acima: em vez de calcular a
    // distancia uma unica vez com cutoff no maior eps, recalcula com cutoff
    // proprio para cada eps -- em dimensao alta, o "early exit" de cada
    // calculo separado compensa o custo de repetir o loop de distancia.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float e0 = eps2_values[0], e1 = eps2_values[1], e2 = eps2_values[2];
    int c0 = 0, c1 = 0, c2 = 0;
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        __syncthreads();
        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, e0, xi_em_reg);
                if (dist <= e0) { c0++; c1++; c2++; continue; }
                dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, e1, xi_em_reg);
                if (dist <= e1) { c1++; c2++; continue; }
                dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, e2, xi_em_reg);
                if (dist <= e2) c2++;
            }
        }
        __syncthreads();
    }
    if (valido) { counts[i] = c0; counts[n + i] = c1; counts[2 * n + i] = c2; }
}

__global__ void count_neighbors_multi_eps_kernel(const float* X, const float* eps2_values, int* counts, int n, int d, int e_count) {
    // Versao generica (numero qualquer de valores de eps), nao so 3.
    // Calcula a distancia uma vez com cutoff no maior eps, acha o primeiro
    // eps em que ela se encaixa, e incrementa os contadores DAQUELE eps
    // em diante (ja que sao encaixados: se serve para eps pequeno, serve
    // para os maiores tambem).
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG;
    bool params_em_reg = e_count <= MAX_PARAM_REG; // se os eps couberem em registrador
    float xi[MAX_D_REG];
    float eps2_reg[MAX_PARAM_REG];
    int local_counts[MAX_PARAM_REG]; // contadores locais em registrador (mais rapido que memoria global)

    if (xi_em_reg) {
        for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    }
    if (params_em_reg) {
        for (int e = 0; e < e_count; e++) {
            eps2_reg[e] = eps2_values[e];
            local_counts[e] = 0;
        }
    }

    float max_eps2 = params_em_reg ? eps2_reg[e_count - 1] : eps2_values[e_count - 1];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) {
            tile_X[k] = X[(long long)tile_start * d + k];
        }
        __syncthreads();

        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue; // fora ate do maior eps, ignora
                int first = -1;
                for (int e = 0; e < e_count; e++) {
                    float eps2 = params_em_reg ? eps2_reg[e] : eps2_values[e];
                    if (dist <= eps2) { first = e; break; } // primeiro eps que "cabe"
                }
                if (first >= 0) {
                    // conta para esse eps e todos os maiores (monotonico)
                    for (int e = first; e < e_count; e++) {
                        if (params_em_reg) local_counts[e]++;
                        else counts[e * n + i]++; // fallback direto em memoria global
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valido && params_em_reg) {
        for (int e = 0; e < e_count; e++) counts[e * n + i] = local_counts[e];
    }
}

__global__ void mark_core_multi_eps_kernel(const int* counts, int* core, int n, int e_count, int min_pts) {
    // Igual ao mark_core_single, mas para 'e_count' grupos de contagem (um por eps).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = e_count * n;
    if (idx < total) core[idx] = (counts[idx] >= min_pts) ? 1 : 0;
}

__global__ void connect_core_multi_eps_kernel(const float* X, const float* eps2_values, const int* core, int* parent, int n, int d, int e_count) {
    // Conecta pares de pontos core, para TODOS os valores de eps de uma vez.
    // Aproveita que "core em eps menor" implica "core em eps maior" para
    // saber, a partir de que indice de eps ('first_e'), a uniao deve
    // acontecer -- e une o par em todos os Union-Finds (um por eps) a
    // partir dali.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    // guarda, para cada ponto do tile, o menor indice de eps em que ele e core
    int* tile_first_core = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_first_core = (i < n) ? first_core_eps_index(core, n, e_count, i) : e_count;
    bool ativo = (i < n && i_first_core < e_count); // so participa se for core em algum eps
    bool xi_em_reg = ativo && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float max_eps2 = eps2_values[e_count - 1];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_first_core[threadIdx.x] = first_core_eps_index(core, n, e_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            int first_local = 0;
            if (tile_start <= i) {
                first_local = i - tile_start + 1; // evita pares duplicados (j > i)
                if (first_local < 0) first_local = 0;
                if (first_local > tile_count) first_local = tile_count;
            }
            for (int local_j = first_local; local_j < tile_count; local_j++) {
                int j_first_core = tile_first_core[local_j];
                if (j_first_core >= e_count) continue; // j nunca e core, ignora
                // o menor eps em que os DOIS (i e j) sao core ao mesmo tempo
                int first_possible = i_first_core > j_first_core ? i_first_core : j_first_core;
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue; // longe demais ate para o maior eps
                int first_e = first_possible;
                // avanca ate achar o primeiro eps grande o suficiente para cobrir a distancia
                while (first_e < e_count && dist > eps2_values[first_e]) first_e++;
                // une nos Union-Finds de first_e ate o ultimo eps
                for (int e = first_e; e < e_count; e++) union_roots_offset(parent, e * n, i, tile_start + local_j);
            }
        }
        __syncthreads();
    }
}

__global__ void connect_core_multi_eps_hidim_kernel(const float* X, const float* eps2_values, const int* core, int* parent, int n, int d, int e_count) {
    // Mesma logica do kernel acima, mas usando 'first_eps_by_distance_repeated_cutoff'
    // (recalcula distancia por eps, com cutoff proprio) -- melhor para alta dimensao.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    int* tile_first_core = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_first_core = (i < n) ? first_core_eps_index(core, n, e_count, i) : e_count;
    bool ativo = (i < n && i_first_core < e_count);
    bool xi_em_reg = ativo && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_first_core[threadIdx.x] = first_core_eps_index(core, n, e_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            int first_local = 0;
            if (tile_start <= i) {
                first_local = i - tile_start + 1;
                if (first_local < 0) first_local = 0;
                if (first_local > tile_count) first_local = tile_count;
            }
            for (int local_j = first_local; local_j < tile_count; local_j++) {
                int j_first_core = tile_first_core[local_j];
                if (j_first_core >= e_count) continue;
                int first_possible = i_first_core > j_first_core ? i_first_core : j_first_core;
                int first_e = first_eps_by_distance_repeated_cutoff(xi, X, tile_X, eps2_values, i, local_j, d, first_possible, e_count, xi_em_reg);
                for (int e = first_e; e < e_count; e++) union_roots_offset(parent, e * n, i, tile_start + local_j);
            }
        }
        __syncthreads();
    }
}



__global__ void assign_border_multi_eps_kernel(const float* X, const float* eps2_values, const int* core, const int* parent, int* labels, int n, int d, int e_count) {
    // Atribui rotulo final para todos os pontos, para TODOS os valores de eps de uma vez.
    // 'remaining' conta quantos grupos (valores de eps) ainda faltam
    // resolver para o ponto i -- quando chega a 0, a thread pode parar
    // de procurar vizinhos (otimizacao, junto com __syncthreads_or).
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    int* tile_first_core = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    int remaining = 0;
    if (valido) {
        for (int e = 0; e < e_count; e++) {
            int base = e * n;
            if (core[base + i]) labels[base + i] = parent[base + i]; // core: rotulo direto
            else { labels[base + i] = -1; remaining++; } // marca como "ruido" ate achar algo melhor
        }
    }
    bool xi_em_reg = valido && remaining > 0 && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float max_eps2 = eps2_values[e_count - 1];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int ativo = (valido && remaining > 0);
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_first_core[threadIdx.x] = first_core_eps_index(core, n, e_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            for (int local_j = 0; local_j < tile_count && remaining > 0; local_j++) {
                int j_first_core = tile_first_core[local_j];
                if (j_first_core >= e_count) continue;
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue;
                int first_e = j_first_core;
                while (first_e < e_count && dist > eps2_values[first_e]) first_e++;
                for (int e = first_e; e < e_count; e++) {
                    int base = e * n;
                    if (labels[base + i] == -1) { // ainda nao resolvido para este eps
                        labels[base + i] = parent[base + tile_start + local_j];
                        remaining--;
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void assign_border_multi_eps_hidim_kernel(const float* X, const float* eps2_values, const int* core, const int* parent, int* labels, int n, int d, int e_count) {
    // Igual ao anterior, mas usando o calculo de distancia repetido por eps
    // (melhor para alta dimensao).
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    int* tile_first_core = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    int remaining = 0;
    if (valido) {
        for (int e = 0; e < e_count; e++) {
            int base = e * n;
            if (core[base + i]) labels[base + i] = parent[base + i];
            else { labels[base + i] = -1; remaining++; }
        }
    }
    bool xi_em_reg = valido && remaining > 0 && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int ativo = (valido && remaining > 0);
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_first_core[threadIdx.x] = first_core_eps_index(core, n, e_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            for (int local_j = 0; local_j < tile_count && remaining > 0; local_j++) {
                int j_first_core = tile_first_core[local_j];
                if (j_first_core >= e_count) continue;
                int first_e = first_eps_by_distance_repeated_cutoff(xi, X, tile_X, eps2_values, i, local_j, d, j_first_core, e_count, xi_em_reg);
                for (int e = first_e; e < e_count; e++) {
                    int base = e * n;
                    if (labels[base + i] == -1) {
                        labels[base + i] = parent[base + tile_start + local_j];
                        remaining--;
                    }
                }
            }
        }
        __syncthreads();
    }
}


// =============================================================================
// KERNELS DO CASO "multi_minpts": eps fixo, varios valores de minPts
// =============================================================================

__global__ void mark_core_multi_minpts_kernel(const int* counts, const int* minpts_values, int* core, int n, int m_count) {
    // Como o eps e fixo, so existe UMA contagem de vizinhos por ponto
    // (nao precisa recontar para cada minPts) -- so compara o mesmo
    // 'counts[i]' contra cada valor de minPts.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m_count * n;
    if (idx >= total) return;
    int m = idx / n;
    int i = idx % n;
    core[idx] = (counts[i] >= minpts_values[m]) ? 1 : 0;
}

__global__ void connect_core_multi_minpts_kernel(const float* X, float eps2, const int* core, int* parent, int n, int d, int m_count) {
    // Conecta pares de pontos core para TODOS os valores de minPts de uma vez.
    // Como minPts e monotonico (quem e core em minPts alto tambem e core em
    // minPts baixo), usa 'core_limit_minpts_index' para saber ate qual indice
    // de minPts cada ponto ainda e core, e une nos Union-Finds correspondentes.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    int* tile_core_limit = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i_limit = (i < n) ? core_limit_minpts_index(core, n, m_count, i) : -1;
    bool ativo = (i < n && i_limit >= 0);
    bool xi_em_reg = ativo && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_core_limit[threadIdx.x] = core_limit_minpts_index(core, n, m_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            int first_local = 0;
            if (tile_start <= i) {
                first_local = i - tile_start + 1;
                if (first_local < 0) first_local = 0;
                if (first_local > tile_count) first_local = tile_count;
            }
            for (int local_j = first_local; local_j < tile_count; local_j++) {
                int j_limit = tile_core_limit[local_j];
                if (j_limit < 0) continue; // j nunca e core para nenhum minPts testado
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, eps2, xi_em_reg);
                if (dist > eps2) continue;
                // o par so pode ser unido nos Union-Finds em que AMBOS sao core,
                // ou seja, ate o menor dos dois limites
                int limit = i_limit < j_limit ? i_limit : j_limit;
                for (int m = 0; m <= limit; m++) union_roots_offset(parent, m * n, i, tile_start + local_j);
            }
        }
        __syncthreads();
    }
}



__global__ void assign_border_multi_minpts_kernel(const float* X, float eps2, const int* core, const int* parent, int* labels, int n, int d, int m_count) {
    // Atribui rotulo final para todos os valores de minPts de uma vez, com a
    // mesma logica de "remaining" usada no caso multi_eps.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    int* tile_core_limit = (int*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    int remaining = 0;
    if (valido) {
        for (int m = 0; m < m_count; m++) {
            int base = m * n;
            if (core[base + i]) labels[base + i] = parent[base + i];
            else { labels[base + i] = -1; remaining++; }
        }
    }
    bool xi_em_reg = valido && remaining > 0 && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int ativo = (valido && remaining > 0);
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_core_limit[threadIdx.x] = core_limit_minpts_index(core, n, m_count, tile_start + threadIdx.x);
        __syncthreads();
        if (ativo) {
            for (int local_j = 0; local_j < tile_count && remaining > 0; local_j++) {
                int limit = tile_core_limit[local_j];
                if (limit < 0) continue;
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, eps2, xi_em_reg);
                if (dist > eps2) continue;
                for (int m = 0; m <= limit; m++) {
                    int base = m * n;
                    if (labels[base + i] == -1) {
                        labels[base + i] = parent[base + tile_start + local_j];
                        remaining--;
                    }
                }
            }
        }
        __syncthreads();
    }
}


// =============================================================================
// KERNELS DO CASO "multi_both": grade combinada de varios eps x varios minPts
// =============================================================================

__global__ void mark_core_multi_eps_minpts_kernel(const int* counts_eps, const int* minpts_values, int* core, int n, int e_count, int m_count) {
    // Para cada combinacao (e, m), marca core comparando a contagem de
    // vizinhos daquele eps ('counts_eps[e*n+point]') contra o minPts 'm'.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = e_count * m_count * n;
    if (idx >= total) return;
    int point = idx % n;
    int combo = idx / n;   // indice da combinacao (e, m) achatado
    int e = combo / m_count;
    int m = combo % m_count;
    core[idx] = (counts_eps[e * n + point] >= minpts_values[m]) ? 1 : 0;
}

__global__ void connect_core_multi_eps_minpts_kernel(const float* X, const float* eps2_values, const int* core, int* parent, int n, int d, int e_count, int m_count) {
    // Versao mais complexa: conecta pares de pontos core para TODAS as
    // combinacoes (eps, minPts) simultaneamente.
    // Usa o "empacotamento" em 64 bits (pack_core_limits_eps_minpts) quando
    // e_count/m_count sao pequenos o suficiente (compact_ok), para evitar
    // reler o array 'core' inteiro a cada comparacao de par.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    unsigned long long* tile_packed_limits = (unsigned long long*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool compact_ok = (e_count <= 16 && m_count <= 15);
    unsigned long long i_packed = (compact_ok && i < n) ? pack_core_limits_eps_minpts(core, n, e_count, m_count, i) : 0ULL;
    // usa a combinacao "mais permissiva" (maior eps, menor minPts = ultimo m_count) como filtro rapido
    int permissive_combo = (e_count - 1) * m_count;
    bool ativo = (i < n && core[permissive_combo * n + i]);
    bool xi_em_reg = ativo && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float max_eps2 = eps2_values[e_count - 1];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_packed_limits[threadIdx.x] = compact_ok ? pack_core_limits_eps_minpts(core, n, e_count, m_count, tile_start + threadIdx.x) : 0ULL;
        __syncthreads();
        if (ativo) {
            int first_local = 0;
            if (tile_start <= i) {
                first_local = i - tile_start + 1;
                if (first_local < 0) first_local = 0;
                if (first_local > tile_count) first_local = tile_count;
            }
            for (int local_j = first_local; local_j < tile_count; local_j++) {
                int j = tile_start + local_j;
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue;
                int first_e = 0;
                while (first_e < e_count && dist > eps2_values[first_e]) first_e++;
                if (compact_ok) {
                    // caminho rapido: le os limites de minPts direto dos inteiros empacotados
                    unsigned long long j_packed = tile_packed_limits[local_j];
                    for (int e = first_e; e < e_count; e++) {
                        int i_limit = unpack_core_limit(i_packed, e);
                        int j_limit = unpack_core_limit(j_packed, e);
                        int limit = i_limit < j_limit ? i_limit : j_limit;
                        for (int m = 0; m <= limit; m++) union_roots_offset(parent, (e * m_count + m) * n, i, j);
                    }
                } else {
                    // caminho generico (fallback): confere 'core' diretamente para cada combinacao
                    for (int e = first_e; e < e_count; e++) {
                        for (int m = 0; m < m_count; m++) {
                            int combo = e * m_count + m;
                            int base = combo * n;
                            if (core[base + i] && core[base + j]) union_roots_offset(parent, base, i, j);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}



__global__ void assign_border_multi_eps_minpts_kernel(const float* X, const float* eps2_values, const int* core, const int* parent, int* labels, int n, int d, int e_count, int m_count) {
    // Atribui rotulo final para TODAS as combinacoes (eps, minPts) de uma vez,
    // usando o mesmo empacotamento em 64 bits quando possivel.
    extern __shared__ unsigned char shared_raw[];
    float* tile_X = (float*)shared_raw;
    unsigned long long* tile_packed_limits = (unsigned long long*)&tile_X[blockDim.x * d];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    int groups = e_count * m_count;
    bool compact_ok = (e_count <= 16 && m_count <= 15);
    int remaining = 0;
    if (valido) {
        for (int combo = 0; combo < groups; combo++) {
            int base = combo * n;
            if (core[base + i]) labels[base + i] = parent[base + i];
            else { labels[base + i] = -1; remaining++; }
        }
    }
    bool xi_em_reg = valido && remaining > 0 && d <= MAX_D_REG;
    float xi[MAX_D_REG];
    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    float max_eps2 = eps2_values[e_count - 1];
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int ativo = (valido && remaining > 0);
        int bloco_tem_trabalho = __syncthreads_or(ativo);
        if (!bloco_tem_trabalho) break;
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) tile_X[k] = X[(long long)tile_start * d + k];
        if (threadIdx.x < tile_count) tile_packed_limits[threadIdx.x] = compact_ok ? pack_core_limits_eps_minpts(core, n, e_count, m_count, tile_start + threadIdx.x) : 0ULL;
        __syncthreads();
        if (ativo) {
            for (int local_j = 0; local_j < tile_count && remaining > 0; local_j++) {
                int j = tile_start + local_j;
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue;
                int first_e = 0;
                while (first_e < e_count && dist > eps2_values[first_e]) first_e++;
                if (compact_ok) {
                    unsigned long long j_packed = tile_packed_limits[local_j];
                    for (int e = first_e; e < e_count; e++) {
                        int limit = unpack_core_limit(j_packed, e);
                        for (int m = 0; m <= limit; m++) {
                            int base = (e * m_count + m) * n;
                            if (labels[base + i] == -1) {
                                labels[base + i] = parent[base + j];
                                remaining--;
                            }
                        }
                    }
                } else {
                    for (int e = first_e; e < e_count; e++) {
                        for (int m = 0; m < m_count; m++) {
                            int base = (e * m_count + m) * n;
                            if (labels[base + i] == -1 && core[base + j]) {
                                labels[base + i] = parent[base + j];
                                remaining--;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}


// =============================================================================
// BACKEND "csr" / "cuml_like"
//
// Inspirado na arquitetura do DBSCAN do cuML:
//   VertexDeg -> CorePoints -> AdjGraph -> ConnectedComponents/MergeLabels -> FinalRelabel.
//
// Esta implementacao NAO usa RAFT/RMM/CUVS nem headers internos do RAPIDS.
// A adaptacao aqui e standalone: primeiro conta graus e monta um CSR para
// eps_max, guardando em first_eps_edge o menor eps que valida cada aresta.
// Depois connect/assign percorrem somente o grafo e nao recalculam distancia.
// Em datasets muito densos, o CSR pode ficar grande; por isso o backend tiled
// antigo continua existindo e pode ser mais adequado quando o grafo explode.
// =============================================================================

__device__ int first_eps_from_dist(float dist, const float* eps2_values, const float* eps2_reg, int e_count, bool params_em_reg) {
    for (int e = 0; e < e_count; e++) {
        float eps2 = params_em_reg ? eps2_reg[e] : eps2_values[e];
        if (dist <= eps2) return e;
    }
    return e_count;
}

__global__ void vertex_degree_multi_eps_kernel(const float* X, const float* eps2_values, int* counts_eps, int* degree_max, int n, int d, int e_count) {
    // Equivalente local do "VertexDeg": conta vizinhos por eps e tambem o grau
    // no maior eps, que vira o tamanho da linha CSR. Inclui i como vizinho de i
    // para manter a semantica de min_samples/minPts do DBSCAN.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG;
    bool params_em_reg = e_count <= MAX_PARAM_REG;
    float xi[MAX_D_REG];
    float eps2_reg[MAX_PARAM_REG];
    int local_counts[MAX_PARAM_REG];

    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    if (params_em_reg) {
        for (int e = 0; e < e_count; e++) {
            eps2_reg[e] = eps2_values[e];
            local_counts[e] = 0;
        }
    } else if (valido) {
        for (int e = 0; e < e_count; e++) counts_eps[e * n + i] = 0;
    }

    float max_eps2 = params_em_reg ? eps2_reg[e_count - 1] : eps2_values[e_count - 1];
    int degree = 0;
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) {
            tile_X[k] = X[(long long)tile_start * d + k];
        }
        __syncthreads();

        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue;
                int first = first_eps_from_dist(dist, eps2_values, eps2_reg, e_count, params_em_reg);
                if (first >= e_count) continue;
                degree++;
                for (int e = first; e < e_count; e++) {
                    if (params_em_reg) local_counts[e]++;
                    else counts_eps[e * n + i]++;
                }
            }
        }
        __syncthreads();
    }

    if (valido) {
        degree_max[i] = degree;
        if (params_em_reg) {
            for (int e = 0; e < e_count; e++) counts_eps[e * n + i] = local_counts[e];
        }
    }
}

__global__ void build_adjgraph_multi_eps_kernel(const float* X, const float* eps2_values, const int* row_ptr, int* col_ind, unsigned char* first_eps_edge, int n, int d, int e_count) {
    // Equivalente local do "AdjGraph": refaz a varredura de distancia uma unica
    // vez para materializar o CSR dentro de eps_max. Depois disso, connect/assign
    // usam apenas col_ind/first_eps_edge.
    extern __shared__ float tile_X[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valido = i < n;
    bool xi_em_reg = valido && d <= MAX_D_REG;
    bool params_em_reg = e_count <= MAX_PARAM_REG;
    float xi[MAX_D_REG];
    float eps2_reg[MAX_PARAM_REG];

    if (xi_em_reg) for (int f = 0; f < d; f++) xi[f] = X[(long long)i * d + f];
    if (params_em_reg) for (int e = 0; e < e_count; e++) eps2_reg[e] = eps2_values[e];

    float max_eps2 = params_em_reg ? eps2_reg[e_count - 1] : eps2_values[e_count - 1];
    int out = valido ? row_ptr[i] : 0;
    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int tile_count = n - tile_start;
        if (tile_count > blockDim.x) tile_count = blockDim.x;
        int tile_values = tile_count * d;
        for (int k = threadIdx.x; k < tile_values; k += blockDim.x) {
            tile_X[k] = X[(long long)tile_start * d + k];
        }
        __syncthreads();

        if (valido) {
            for (int local_j = 0; local_j < tile_count; local_j++) {
                float dist = dist_sq_tile_cutoff(xi, X, tile_X, i, local_j, d, max_eps2, xi_em_reg);
                if (dist > max_eps2) continue;
                int first = first_eps_from_dist(dist, eps2_values, eps2_reg, e_count, params_em_reg);
                if (first >= e_count) continue;
                col_ind[out] = tile_start + local_j;
                first_eps_edge[out] = (unsigned char)first;
                out++;
            }
        }
        __syncthreads();
    }
}

__global__ void connect_core_csr_multi_eps_kernel(const int* row_ptr, const int* col_ind, const unsigned char* first_eps_edge, const int* core, int* parent, int n, int e_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int i_first_core = first_core_eps_index(core, n, e_count, i);
    if (i_first_core >= e_count) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        int j = col_ind[p];
        if (j <= i) continue; // CSR e direcionado completo; evita unir o mesmo par duas vezes e ignora i==j.
        int j_first_core = first_core_eps_index(core, n, e_count, j);
        if (j_first_core >= e_count) continue;
        int first_e = (int)first_eps_edge[p];
        if (first_e < i_first_core) first_e = i_first_core;
        if (first_e < j_first_core) first_e = j_first_core;
        for (int e = first_e; e < e_count; e++) union_roots_offset(parent, e * n, i, j);
    }
}

__global__ void connect_core_csr_multi_minpts_kernel(const int* row_ptr, const int* col_ind, const int* core, int* parent, int n, int m_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int i_limit = core_limit_minpts_index(core, n, m_count, i);
    if (i_limit < 0) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        int j = col_ind[p];
        if (j <= i) continue;
        int j_limit = core_limit_minpts_index(core, n, m_count, j);
        if (j_limit < 0) continue;
        int limit = i_limit < j_limit ? i_limit : j_limit;
        for (int m = 0; m <= limit; m++) union_roots_offset(parent, m * n, i, j);
    }
}

__global__ void connect_core_csr_multi_both_kernel(const int* row_ptr, const int* col_ind, const unsigned char* first_eps_edge, const int* core, int* parent, int n, int e_count, int m_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int permissive_combo = (e_count - 1) * m_count; // maior eps, menor minPts
    if (!core[permissive_combo * n + i]) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        int j = col_ind[p];
        if (j <= i) continue;
        if (!core[permissive_combo * n + j]) continue;
        int first_e = (int)first_eps_edge[p];
        for (int e = first_e; e < e_count; e++) {
            for (int m = 0; m < m_count; m++) {
                int combo = e * m_count + m;
                int base = combo * n;
                if (core[base + i] && core[base + j]) union_roots_offset(parent, base, i, j);
            }
        }
    }
}

__global__ void assign_border_csr_multi_eps_kernel(const int* row_ptr, const int* col_ind, const unsigned char* first_eps_edge, const int* core, const int* parent, int* labels, int n, int e_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = e_count * n;
    if (idx >= total) return;
    int e = idx / n;
    int i = idx % n;
    int base = e * n;
    if (core[base + i]) {
        labels[base + i] = parent[base + i];
        return;
    }
    int best = INT_MAX;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        if (e < (int)first_eps_edge[p]) continue;
        int j = col_ind[p];
        if (core[base + j]) {
            int root = parent[base + j];
            if (root < best) best = root;
        }
    }
    labels[base + i] = (best == INT_MAX) ? -1 : best;
}

__global__ void assign_border_csr_multi_minpts_kernel(const int* row_ptr, const int* col_ind, const int* core, const int* parent, int* labels, int n, int m_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m_count * n;
    if (idx >= total) return;
    int m = idx / n;
    int i = idx % n;
    int base = m * n;
    if (core[base + i]) {
        labels[base + i] = parent[base + i];
        return;
    }
    int best = INT_MAX;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        int j = col_ind[p];
        if (core[base + j]) {
            int root = parent[base + j];
            if (root < best) best = root;
        }
    }
    labels[base + i] = (best == INT_MAX) ? -1 : best;
}

__global__ void assign_border_csr_multi_both_kernel(const int* row_ptr, const int* col_ind, const unsigned char* first_eps_edge, const int* core, const int* parent, int* labels, int n, int e_count, int m_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int groups = e_count * m_count;
    int total = groups * n;
    if (idx >= total) return;
    int combo = idx / n;
    int i = idx % n;
    int e = combo / m_count;
    int base = combo * n;
    if (core[base + i]) {
        labels[base + i] = parent[base + i];
        return;
    }
    int best = INT_MAX;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        if (e < (int)first_eps_edge[p]) continue;
        int j = col_ind[p];
        if (core[base + j]) {
            int root = parent[base + j];
            if (root < best) best = root;
        }
    }
    labels[base + i] = (best == INT_MAX) ? -1 : best;
}


// =============================================================================
// FUNCOES AUXILIARES DE I/O (leitura/escrita de arquivos binarios) E ESTATISTICAS
// =============================================================================

std::vector<float> read_f32(const char* path, long long expected) {
    // Le um arquivo binario de floats (o dataset X, ja em formato bruto),
    // e confere que o tamanho lido bate com o esperado (n * d floats).
    std::ifstream in(path, std::ios::binary);
    if (!in) { fprintf(stderr, "nao abriu input\n"); exit(2); }
    std::vector<float> data(expected);
    in.read((char*)data.data(), expected * sizeof(float));
    if (in.gcount() != expected * (long long)sizeof(float)) { fprintf(stderr, "input com tamanho invalido\n"); exit(3); }
    return data;
}

void write_i32(const char* path, const std::vector<int>& data) {
    // Escreve um vetor de inteiros em binario (usado para labels e core).
    std::ofstream out(path, std::ios::binary);
    out.write((const char*)data.data(), data.size() * sizeof(int));
}

int count_clusters(const int* labels, int n) {
    // Conta quantos clusters distintos existem (ignora -1, que e ruido).
    std::unordered_set<int> s;
    for (int i = 0; i < n; i++) if (labels[i] != -1) s.insert(labels[i]);
    return (int)s.size();
}

void print_stats(const std::vector<int>& labels, const std::vector<int>& core, int groups, int n) {
    // Imprime, para cada "grupo" (combinacao de parametros), quantos
    // clusters foram formados, quantos pontos ficaram como ruido e
    // quantos pontos sao core.
    for (int g = 0; g < groups; g++) {
        int noise = 0, cores = 0;
        for (int i = 0; i < n; i++) {
            if (labels[g*n+i] == -1) noise++;
            if (core[g*n+i]) cores++;
        }
        printf("group_%d_clusters=%d group_%d_noise=%d group_%d_core=%d ", g, count_clusters(labels.data() + g*n, n), g, noise, g, cores);
    }
}

void compact_relabel_cpu(std::vector<int>& labels, int groups, int n) {
    // Opcionalmente aproxima a etapa FinalRelabel do cuML: preserva ruido -1 e
    // troca raizes arbitrarias por IDs compactos 0..K-1 em ordem de aparicao.
    for (int g = 0; g < groups; g++) {
        std::unordered_map<int, int> map_root_to_compact;
        int next_label = 0;
        int base = g * n;
        for (int i = 0; i < n; i++) {
            int lbl = labels[base + i];
            if (lbl == -1) continue;
            auto it = map_root_to_compact.find(lbl);
            if (it == map_root_to_compact.end()) {
                it = map_root_to_compact.emplace(lbl, next_label++).first;
            }
            labels[base + i] = it->second;
        }
    }
}

float elapsed(cudaEvent_t a, cudaEvent_t b) { float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); return ms; }


// -----------------------------------------------------------------------
// Estrutura que agrupa todos os ponteiros de GPU e tempos medidos de uma
// execucao completa do pipeline (contagem -> core -> conexao -> flatten -> borda)
// -----------------------------------------------------------------------
struct ExecResult {
    int groups = 1;              // numero de combinacoes de parametros (1 para "seq")
    int* counts = nullptr;
    int* core = nullptr;
    int* parent = nullptr;
    int* labels = nullptr;
    int* minpts = nullptr;
    float* eps2_values = nullptr;
    float event_count_ms = 0.0f;
    float event_mark_ms = 0.0f;
    float event_connect_ms = 0.0f;
    float event_flatten_ms = 0.0f;
    float event_assign_ms = 0.0f;
    float event_total_ms = 0.0f;
    float event_degree_ms = 0.0f;
    float event_prefix_ms = 0.0f;
    float event_adj_ms = 0.0f;
    float event_connect_graph_ms = 0.0f;
    float event_assign_graph_ms = 0.0f;
    long long nnz = 0;
    double avg_degree = 0.0;
    double graph_mbytes = 0.0;
    int fallback_used = 0;
    std::string fallback_reason = "none";
    std::string backend = "tiled";
};

void free_exec_result(ExecResult& r) {
    // Libera toda a memoria de GPU alocada nesta execucao.
    if (r.counts) CUDA_CHECK(cudaFree(r.counts));
    if (r.core) CUDA_CHECK(cudaFree(r.core));
    if (r.parent) CUDA_CHECK(cudaFree(r.parent));
    if (r.labels) CUDA_CHECK(cudaFree(r.labels));
    if (r.minpts) CUDA_CHECK(cudaFree(r.minpts));
    if (r.eps2_values) CUDA_CHECK(cudaFree(r.eps2_values));
    r.counts = nullptr;
    r.core = nullptr;
    r.parent = nullptr;
    r.labels = nullptr;
    r.minpts = nullptr;
    r.eps2_values = nullptr;
}


// =============================================================================
// FUNCAO PRINCIPAL
//
// Uso: <mode> <input> <labels_out> <core_out> <n> <d> <parametros especificos do mode>
//   mode = seq         -> ... n d eps min_pts
//   mode = multi_eps    -> ... n d min_pts e_count eps0 eps1 ... epsK
//   mode = multi_minpts -> ... n d eps m_count minpts0 minpts1 ... minptsK
//   mode = multi_both    -> ... n d e_count m_count eps0..epsE minpts0..minptsM
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 9) { fprintf(stderr, "uso: mode input labels core n d ...\n"); return 1; }
    std::string mode = argv[1];
    const char* input_path = argv[2];
    const char* labels_path = argv[3];
    const char* core_path = argv[4];
    int n = atoi(argv[5]);
    int d = atoi(argv[6]);
    int threads = 256; // tamanho inicial do bloco (threads por bloco)
    std::string backend_requested = "tiled";
    double max_graph_mbytes = DEFAULT_MAX_GRAPH_MBYTES;
    int final_relabel = 0;

    // Flags opcionais devem vir depois dos parametros posicionais do modo.
    // Exemplos:
    //   ... multi_eps ... eps0 eps1 eps2 --backend csr --max_graph_mbytes 1024
    //   ... seq ... eps minPts --backend tiled
    for (int ai = 7; ai < argc; ai++) {
        std::string arg = argv[ai];
        if (arg == "--backend" && ai + 1 < argc) {
            backend_requested = argv[++ai];
        } else if (arg == "--max_graph_mbytes" && ai + 1 < argc) {
            max_graph_mbytes = atof(argv[++ai]);
        } else if (arg == "--final_relabel" && ai + 1 < argc) {
            final_relabel = atoi(argv[++ai]);
        }
    }
    if (backend_requested == "cuml_like") backend_requested = "csr";
    if (backend_requested != "tiled" && backend_requested != "csr") {
        fprintf(stderr, "backend invalido: %s (usando tiled)\n", backend_requested.c_str());
        backend_requested = "tiled";
    }

    // Consulta as propriedades da GPU para saber quanta shared memory por bloco existe,
    // e reduz o numero de threads por bloco ate que o tile caiba nessa memoria.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t shared_tile_bytes = (size_t)threads * d * sizeof(float);                                  // so as coordenadas do tile
    size_t shared_int_tile_bytes = shared_tile_bytes + (size_t)threads * sizeof(int);                // + vetor auxiliar de inteiros
    size_t shared_u64_tile_bytes = shared_tile_bytes + (size_t)threads * sizeof(unsigned long long); // + vetor auxiliar de 64 bits (o "pior caso")
    while (shared_u64_tile_bytes > (size_t)prop.sharedMemPerBlock && threads > 32) {
        threads /= 2; // reduz threads por bloco pela metade ate caber
        shared_tile_bytes = (size_t)threads * d * sizeof(float);
        shared_int_tile_bytes = shared_tile_bytes + (size_t)threads * sizeof(int);
        shared_u64_tile_bytes = shared_tile_bytes + (size_t)threads * sizeof(unsigned long long);
    }
    if (shared_u64_tile_bytes > (size_t)prop.sharedMemPerBlock) {
        fprintf(stderr, "shared memory insuficiente para d=%d e tile=%d\n", d, threads);
        return 5;
    }
    int blocks = (n + threads - 1) / threads; // numero de blocos necessarios para cobrir todos os n pontos

    // Le o dataset do arquivo binario e copia para a GPU
    std::vector<float> h_X = read_f32(input_path, (long long)n * d);
    float* d_X = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, h_X.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Eventos CUDA para medir o tempo de cada etapa do pipeline
    cudaEvent_t t0,t1,t2,t3,t4,t5,t6,t7;
    cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);
    cudaEventCreate(&t4); cudaEventCreate(&t5); cudaEventCreate(&t6); cudaEventCreate(&t7);

    // Decide, so para fins de log/diagnostico, qual "estrategia" de calculo
    // de distancia sera usada (normal vs hidim), baseado em dimensao e
    // razao entre o maior e o menor eps.
    std::string eps_strategy = "na";
    if (mode == "multi_eps") {
        int e_count_preview = atoi(argv[8]);
        float first_eps = atof(argv[9]);
        float last_eps = atof(argv[9 + e_count_preview - 1]);
        eps_strategy = (d >= 16 && e_count_preview == 3 && last_eps * last_eps > 4.0f * first_eps * first_eps) ? "hidim_per_eps_cutoff" : "max_eps_reuse";
    } else if (mode == "multi_both") {
        int e_count_preview = atoi(argv[7]);
        float first_eps = atof(argv[9]);
        float last_eps = atof(argv[9 + e_count_preview - 1]);
        eps_strategy = (d >= 16 && e_count_preview == 3 && last_eps * last_eps > 4.0f * first_eps * first_eps) ? "hidim_count_eps3" : "max_eps_reuse";
    }

    // Lambda que executa o pipeline completo (todas as etapas), de acordo
    // com o 'mode' escolhido. 'medir=true' registra os eventos de tempo;
    // 'medir=false' e usado so para o warm-up (aquecimento da GPU).
    auto executar_tiled = [&](bool medir) -> ExecResult {
        ExecResult r;
        r.backend = "tiled";
        auto record = [&](cudaEvent_t ev) { if (medir) CUDA_CHECK(cudaEventRecord(ev)); };

        if (mode == "seq") {
            // -------- caso simples: um eps e um minPts --------
            float eps = atof(argv[7]);
            int min_pts = atoi(argv[8]);
            float eps2 = eps * eps;
            CUDA_CHECK(cudaMalloc(&r.counts, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.core, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.parent, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.labels, n * sizeof(int)));
            record(t0);
            init_parent_kernel<<<blocks, threads>>>(r.parent, 1, n);
            CUDA_CHECK(cudaMemset(r.counts, 0, n * sizeof(int)));
            count_neighbors_single_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.counts, n, d, eps2); record(t1);
            mark_core_single_kernel<<<blocks, threads>>>(r.counts, r.core, n, min_pts); record(t2);
            connect_cores_single_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.core, r.parent, n, d, eps2); record(t3);
            flatten_parent_kernel<<<blocks, threads>>>(r.parent, 1, n); record(t4);
            assign_borders_single_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.core, r.parent, r.labels, n, d, eps2); record(t5);
        } else if (mode == "multi_eps") {
            // -------- varios eps, minPts fixo --------
            int min_pts = atoi(argv[7]);
            int e_count = atoi(argv[8]);
            r.groups = e_count;
            std::vector<float> eps2(e_count);
            for (int e = 0; e < e_count; e++) { float eps = atof(argv[9+e]); eps2[e] = eps * eps; }
            CUDA_CHECK(cudaMalloc(&r.eps2_values, e_count * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(r.eps2_values, eps2.data(), e_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&r.counts, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMemset(r.counts, 0, r.groups*n*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.core, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.parent, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.labels, r.groups*n*sizeof(int)));
            record(t0);
            init_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n);
            // escolhe entre o kernel especializado de 3 eps (normal ou hidim) ou o generico
            bool use_hidim_eps = (d >= 16 && e_count == 3 && eps2[e_count - 1] > 4.0f * eps2[0]);
            if (e_count == 3 && use_hidim_eps) count_neighbors_multi_eps3_hidim_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d);
            else if (e_count == 3) count_neighbors_multi_eps3_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d);
            else count_neighbors_multi_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d, e_count);
            record(t1);
            mark_core_multi_eps_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.counts, r.core, n, e_count, min_pts); record(t2);
            if (use_hidim_eps) connect_core_multi_eps_hidim_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, n, d, e_count);
            else connect_core_multi_eps_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, n, d, e_count);
            record(t3);
            flatten_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n); record(t4);
            if (use_hidim_eps) assign_border_multi_eps_hidim_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, r.labels, n, d, e_count);
            else assign_border_multi_eps_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, r.labels, n, d, e_count);
            record(t5);
        } else if (mode == "multi_minpts") {
            // -------- eps fixo, varios minPts --------
            float eps = atof(argv[7]);
            int m_count = atoi(argv[8]);
            r.groups = m_count;
            float eps2 = eps * eps;
            std::vector<int> minpts(m_count);
            for (int m = 0; m < m_count; m++) minpts[m] = atoi(argv[9+m]);
            CUDA_CHECK(cudaMalloc(&r.minpts, m_count * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(r.minpts, minpts.data(), m_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&r.counts, n*sizeof(int))); CUDA_CHECK(cudaMemset(r.counts, 0, n*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.core, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.parent, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.labels, r.groups*n*sizeof(int)));
            record(t0);
            init_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n);
            // so precisa contar vizinhos UMA vez, ja que o eps e o mesmo para todos os minPts
            count_neighbors_single_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.counts, n, d, eps2); record(t1);
            mark_core_multi_minpts_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.counts, r.minpts, r.core, n, m_count); record(t2);
            connect_core_multi_minpts_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, eps2, r.core, r.parent, n, d, m_count); record(t3);
            flatten_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n); record(t4);
            assign_border_multi_minpts_kernel<<<blocks, threads, shared_int_tile_bytes>>>(d_X, eps2, r.core, r.parent, r.labels, n, d, m_count); record(t5);
        } else if (mode == "multi_both") {
            // -------- grade combinada: varios eps x varios minPts --------
            int e_count = atoi(argv[7]);
            int m_count = atoi(argv[8]);
            r.groups = e_count * m_count;
            std::vector<float> eps2(e_count);
            for (int e = 0; e < e_count; e++) { float eps = atof(argv[9 + e]); eps2[e] = eps * eps; }
            std::vector<int> minpts(m_count);
            for (int m = 0; m < m_count; m++) minpts[m] = atoi(argv[9 + e_count + m]);
            CUDA_CHECK(cudaMalloc(&r.eps2_values, e_count * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(r.eps2_values, eps2.data(), e_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&r.minpts, m_count * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(r.minpts, minpts.data(), m_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&r.counts, e_count*n*sizeof(int))); CUDA_CHECK(cudaMemset(r.counts, 0, e_count*n*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&r.core, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.parent, r.groups*n*sizeof(int))); CUDA_CHECK(cudaMalloc(&r.labels, r.groups*n*sizeof(int)));
            record(t0);
            init_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n);
            // conta vizinhos so por eps (nao por combinacao eps x minPts) e reaproveita para todos os minPts
            bool use_hidim_eps = (d >= 16 && e_count == 3 && eps2[e_count - 1] > 4.0f * eps2[0]);
            if (e_count == 3 && use_hidim_eps) count_neighbors_multi_eps3_hidim_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d);
            else if (e_count == 3) count_neighbors_multi_eps3_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d);
            else count_neighbors_multi_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, n, d, e_count);
            record(t1);
            mark_core_multi_eps_minpts_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.counts, r.minpts, r.core, n, e_count, m_count); record(t2);
            connect_core_multi_eps_minpts_kernel<<<blocks, threads, shared_u64_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, n, d, e_count, m_count); record(t3);
            flatten_parent_kernel<<<(r.groups*n+threads-1)/threads, threads>>>(r.parent, r.groups, n); record(t4);
            assign_border_multi_eps_minpts_kernel<<<blocks, threads, shared_u64_tile_bytes>>>(d_X, r.eps2_values, r.core, r.parent, r.labels, n, d, e_count, m_count); record(t5);
        } else { fprintf(stderr, "mode invalido\n"); exit(4); }

        CUDA_CHECK(cudaGetLastError());
        if (medir) {
            CUDA_CHECK(cudaEventSynchronize(t5));
            r.event_count_ms = elapsed(t0,t1);
            r.event_mark_ms = elapsed(t1,t2);
            r.event_connect_ms = elapsed(t2,t3);
            r.event_flatten_ms = elapsed(t3,t4);
            r.event_assign_ms = elapsed(t4,t5);
            r.event_total_ms = elapsed(t0,t5);
            r.event_degree_ms = r.event_count_ms;
            r.event_connect_graph_ms = r.event_connect_ms;
            r.event_assign_graph_ms = r.event_assign_ms;
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        return r;
    };

    auto executar_csr = [&](bool medir) -> ExecResult {
        ExecResult r;
        r.backend = "csr";
        auto record = [&](cudaEvent_t ev) { if (medir) CUDA_CHECK(cudaEventRecord(ev)); };

        int e_count = 1;
        int m_count = 1;
        int min_pts = 0;
        enum ModeKind { MODE_SEQ, MODE_MULTI_EPS, MODE_MULTI_MINPTS, MODE_MULTI_BOTH };
        ModeKind kind = MODE_SEQ;
        std::vector<float> eps2;
        std::vector<int> minpts;

        if (mode == "seq") {
            kind = MODE_SEQ;
            float eps = atof(argv[7]);
            min_pts = atoi(argv[8]);
            eps2.push_back(eps * eps);
            r.groups = 1;
        } else if (mode == "multi_eps") {
            kind = MODE_MULTI_EPS;
            min_pts = atoi(argv[7]);
            e_count = atoi(argv[8]);
            r.groups = e_count;
            eps2.resize(e_count);
            for (int e = 0; e < e_count; e++) {
                float eps = atof(argv[9 + e]);
                eps2[e] = eps * eps;
            }
        } else if (mode == "multi_minpts") {
            kind = MODE_MULTI_MINPTS;
            float eps = atof(argv[7]);
            e_count = 1;
            m_count = atoi(argv[8]);
            r.groups = m_count;
            eps2.push_back(eps * eps);
            minpts.resize(m_count);
            for (int m = 0; m < m_count; m++) minpts[m] = atoi(argv[9 + m]);
        } else if (mode == "multi_both") {
            kind = MODE_MULTI_BOTH;
            e_count = atoi(argv[7]);
            m_count = atoi(argv[8]);
            r.groups = e_count * m_count;
            eps2.resize(e_count);
            for (int e = 0; e < e_count; e++) {
                float eps = atof(argv[9 + e]);
                eps2[e] = eps * eps;
            }
            minpts.resize(m_count);
            for (int m = 0; m < m_count; m++) minpts[m] = atoi(argv[9 + e_count + m]);
        } else {
            fprintf(stderr, "mode invalido\n");
            exit(4);
        }

        int* degree_max = nullptr;
        int* row_ptr = nullptr;
        int* col_ind = nullptr;
        unsigned char* first_eps_edge = nullptr;

        auto cleanup_graph = [&]() {
            if (degree_max) CUDA_CHECK(cudaFree(degree_max));
            if (row_ptr) CUDA_CHECK(cudaFree(row_ptr));
            if (col_ind) CUDA_CHECK(cudaFree(col_ind));
            if (first_eps_edge) CUDA_CHECK(cudaFree(first_eps_edge));
            degree_max = nullptr;
            row_ptr = nullptr;
            col_ind = nullptr;
            first_eps_edge = nullptr;
        };

        auto fallback_return = [&](const std::string& reason, long long attempted_nnz, double attempted_mbytes) -> ExecResult {
            cleanup_graph();
            free_exec_result(r);
            ExecResult f = executar_tiled(medir);
            f.fallback_used = 1;
            f.fallback_reason = reason;
            f.backend = "tiled";
            f.nnz = attempted_nnz;
            f.avg_degree = (n > 0) ? (double)attempted_nnz / (double)n : 0.0;
            f.graph_mbytes = attempted_mbytes;
            return f;
        };

        if (e_count <= 0 || r.groups <= 0) return fallback_return("invalid_param_count", 0, 0.0);
        if (e_count > 255) return fallback_return("too_many_eps_for_uchar", 0, 0.0);

        long long counts_elems = (long long)e_count * n;
        long long group_elems = (long long)r.groups * n;

        CUDA_CHECK(cudaMalloc(&r.eps2_values, e_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(r.eps2_values, eps2.data(), e_count * sizeof(float), cudaMemcpyHostToDevice));
        if (kind == MODE_MULTI_MINPTS || kind == MODE_MULTI_BOTH) {
            CUDA_CHECK(cudaMalloc(&r.minpts, m_count * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(r.minpts, minpts.data(), m_count * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&r.counts, counts_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&r.core, group_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&r.parent, group_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&r.labels, group_elems * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&degree_max, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&row_ptr, ((long long)n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(r.counts, 0, counts_elems * sizeof(int)));

        record(t0);
        init_parent_kernel<<<(group_elems + threads - 1) / threads, threads>>>(r.parent, r.groups, n);
        vertex_degree_multi_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, r.counts, degree_max, n, d, e_count);
        record(t1);

        thrust::device_ptr<int> degree_ptr = thrust::device_pointer_cast(degree_max);
        long long nnz_ll = thrust::reduce(thrust::device, degree_ptr, degree_ptr + n, 0LL, thrust::plus<long long>());
        r.nnz = nnz_ll;
        r.avg_degree = (n > 0) ? (double)nnz_ll / (double)n : 0.0;
        double graph_mbytes = ((double)((long long)n + 1) * sizeof(int) + (double)nnz_ll * (sizeof(int) + sizeof(unsigned char))) / (1024.0 * 1024.0);
        r.graph_mbytes = graph_mbytes;

        if (nnz_ll > (long long)INT_MAX) return fallback_return("nnz_over_int", nnz_ll, graph_mbytes);
        if (max_graph_mbytes > 0.0 && graph_mbytes > max_graph_mbytes) return fallback_return("graph_limit_mbytes", nnz_ll, graph_mbytes);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        double graph_bytes = (double)((long long)n + 1) * sizeof(int) + (double)nnz_ll * (sizeof(int) + sizeof(unsigned char));
        if (graph_bytes > (double)free_bytes * 0.85) return fallback_return("insufficient_free_mem", nnz_ll, graph_mbytes);

        CUDA_CHECK(cudaMemset(row_ptr, 0, sizeof(int)));
        thrust::device_ptr<int> row_ptr_plus_one = thrust::device_pointer_cast(row_ptr + 1);
        thrust::inclusive_scan(thrust::device, degree_ptr, degree_ptr + n, row_ptr_plus_one);
        record(t2);

        cudaError_t malloc_err = cudaMalloc(&col_ind, nnz_ll * sizeof(int));
        if (malloc_err != cudaSuccess) {
            cudaGetLastError();
            return fallback_return("cudaMalloc_col_ind", nnz_ll, graph_mbytes);
        }
        malloc_err = cudaMalloc(&first_eps_edge, nnz_ll * sizeof(unsigned char));
        if (malloc_err != cudaSuccess) {
            cudaGetLastError();
            return fallback_return("cudaMalloc_first_eps_edge", nnz_ll, graph_mbytes);
        }

        build_adjgraph_multi_eps_kernel<<<blocks, threads, shared_tile_bytes>>>(d_X, r.eps2_values, row_ptr, col_ind, first_eps_edge, n, d, e_count);
        record(t3);

        if (kind == MODE_SEQ) {
            mark_core_single_kernel<<<blocks, threads>>>(r.counts, r.core, n, min_pts);
        } else if (kind == MODE_MULTI_EPS) {
            mark_core_multi_eps_kernel<<<(group_elems + threads - 1) / threads, threads>>>(r.counts, r.core, n, e_count, min_pts);
        } else if (kind == MODE_MULTI_MINPTS) {
            mark_core_multi_minpts_kernel<<<(group_elems + threads - 1) / threads, threads>>>(r.counts, r.minpts, r.core, n, m_count);
        } else {
            mark_core_multi_eps_minpts_kernel<<<(group_elems + threads - 1) / threads, threads>>>(r.counts, r.minpts, r.core, n, e_count, m_count);
        }
        record(t4);

        if (kind == MODE_SEQ) {
            connect_core_csr_multi_minpts_kernel<<<blocks, threads>>>(row_ptr, col_ind, r.core, r.parent, n, 1);
        } else if (kind == MODE_MULTI_EPS) {
            connect_core_csr_multi_eps_kernel<<<blocks, threads>>>(row_ptr, col_ind, first_eps_edge, r.core, r.parent, n, e_count);
        } else if (kind == MODE_MULTI_MINPTS) {
            connect_core_csr_multi_minpts_kernel<<<blocks, threads>>>(row_ptr, col_ind, r.core, r.parent, n, m_count);
        } else {
            connect_core_csr_multi_both_kernel<<<blocks, threads>>>(row_ptr, col_ind, first_eps_edge, r.core, r.parent, n, e_count, m_count);
        }
        record(t5);

        flatten_parent_kernel<<<(group_elems + threads - 1) / threads, threads>>>(r.parent, r.groups, n);
        record(t6);

        if (kind == MODE_SEQ) {
            assign_border_csr_multi_minpts_kernel<<<blocks, threads>>>(row_ptr, col_ind, r.core, r.parent, r.labels, n, 1);
        } else if (kind == MODE_MULTI_EPS) {
            assign_border_csr_multi_eps_kernel<<<(group_elems + threads - 1) / threads, threads>>>(row_ptr, col_ind, first_eps_edge, r.core, r.parent, r.labels, n, e_count);
        } else if (kind == MODE_MULTI_MINPTS) {
            assign_border_csr_multi_minpts_kernel<<<(group_elems + threads - 1) / threads, threads>>>(row_ptr, col_ind, r.core, r.parent, r.labels, n, m_count);
        } else {
            assign_border_csr_multi_both_kernel<<<(group_elems + threads - 1) / threads, threads>>>(row_ptr, col_ind, first_eps_edge, r.core, r.parent, r.labels, n, e_count, m_count);
        }
        record(t7);

        CUDA_CHECK(cudaGetLastError());
        if (medir) {
            CUDA_CHECK(cudaEventSynchronize(t7));
            r.event_degree_ms = elapsed(t0, t1);
            r.event_prefix_ms = elapsed(t1, t2);
            r.event_adj_ms = elapsed(t2, t3);
            r.event_mark_ms = elapsed(t3, t4);
            r.event_connect_graph_ms = elapsed(t4, t5);
            r.event_flatten_ms = elapsed(t5, t6);
            r.event_assign_graph_ms = elapsed(t6, t7);
            r.event_total_ms = elapsed(t0, t7);
            r.event_count_ms = r.event_degree_ms;
            r.event_connect_ms = r.event_connect_graph_ms;
            r.event_assign_ms = r.event_assign_graph_ms;
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cleanup_graph();
        return r;
    };

    auto executar = [&](bool medir) -> ExecResult {
        if (backend_requested == "csr") return executar_csr(medir);
        return executar_tiled(medir);
    };

    // Warm-up real do CUDA proprio: mesma sequencia de kernels, mesmo processo, tempo descartado.
    // Isso evita que o custo de "aquecer" a GPU (primeira chamada de kernel,
    // compilacao JIT, etc.) polua a medicao de tempo real.
    ExecResult warm = executar(false);
    free_exec_result(warm);

    // Executa de verdade, agora medindo o tempo com cudaProfilerStart/Stop
    // (para permitir profiling externo via nsys/nvprof, se usado).
    CUDA_CHECK(cudaProfilerStart());
    ExecResult measured = executar(true);
    CUDA_CHECK(cudaProfilerStop());

    // Copia os resultados (labels e core) de volta para a CPU e grava em disco
    std::vector<int> h_labels(measured.groups*n), h_core(measured.groups*n);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), measured.labels, measured.groups*n*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_core.data(), measured.core, measured.groups*n*sizeof(int), cudaMemcpyDeviceToHost));
    if (final_relabel) compact_relabel_cpu(h_labels, measured.groups, n);
    write_i32(labels_path, h_labels);
    write_i32(core_path, h_core);

    // Imprime um resumo em uma unica linha (formato "chave=valor"), facil de
    // ser parseado por um script Python que rode este binario e capture o stdout
    printf("mode=%s groups=%d n=%d d=%d backend=%s requested_backend=%s fallback_used=%d fallback_reason=%s cuda_warmup=1 count_kernel=%s shared_tile_points=%d shared_tile_bytes=%lld max_d_reg=%d max_param_reg=%d max_graph_mbytes=%.3f nnz=%lld avg_degree=%.6f graph_mbytes=%.6f event_degree_ms=%.6f event_prefix_ms=%.6f event_adj_ms=%.6f event_mark_ms=%.6f event_connect_graph_ms=%.6f event_flatten_ms=%.6f event_assign_graph_ms=%.6f event_count_ms=%.6f event_connect_ms=%.6f event_assign_ms=%.6f event_total_ms=%.6f total_ms=%.6f eps_strategy=%s shared_int_tile_bytes=%lld shared_u64_tile_bytes=%lld final_relabel=%d ",
           mode.c_str(), measured.groups, n, d, measured.backend.c_str(), backend_requested.c_str(), measured.fallback_used, measured.fallback_reason.c_str(), measured.backend == "csr" ? "csr_epsmax" : "shared_tiled", threads, (long long)shared_tile_bytes, MAX_D_REG, MAX_PARAM_REG, max_graph_mbytes, measured.nnz, measured.avg_degree, measured.graph_mbytes, measured.event_degree_ms, measured.event_prefix_ms, measured.event_adj_ms, measured.event_mark_ms, measured.event_connect_graph_ms, measured.event_flatten_ms, measured.event_assign_graph_ms, measured.event_count_ms, measured.event_connect_ms, measured.event_assign_ms, measured.event_total_ms, measured.event_total_ms, eps_strategy.c_str(), (long long)shared_int_tile_bytes, (long long)shared_u64_tile_bytes, final_relabel);
    print_stats(h_labels, h_core, measured.groups, n);
    printf("\n");

    // Libera toda a memoria alocada na GPU
    free_exec_result(measured);
    CUDA_CHECK(cudaFree(d_X));
    return 0;
}
