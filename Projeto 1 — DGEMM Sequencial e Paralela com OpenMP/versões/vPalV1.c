/*****************************************************************************************
 * *
 * Título: Projeto 1 — DGEMM Sequencial e Paralela com OpenMP                             *
 * Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves                      *
 * Disciplina: DEC107 — Processamento Paralelo                                           *
 * Data: 29/09/2025                                                                      *
 * Arquivo: vPalV1.c                                                                     *
 * *
 * Objetivo do Projeto:                                                                  *
 * Implementar e avaliar o desempenho de rotinas de multiplicação de matrizes            *
 * (DGEMM) em versões sequencial e paralela (com OpenMP), aplicando conceitos de         *
 * arquitetura de memória compartilhada e análise de performance.                        *
 * *
 * Como Compilar:                                                                        *
 * gcc -O3 -fopenmp -o dgemm_teste vPalV1.c -lopenblas                                   *
 * *
 *****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>

// --- Protótipos das Funções ---
double* alocar_matriz(int dimensao);
void inicializar_matrizes(double *A, double *B, double *C, int dimensao);
void dgemm_sequencial(double *A, double *B, double *C, int dimensao);
void dgemm_paralelo(double *A, double *B, double *C, int dimensao);

// --- Função Principal ---
int main(void) {
    // --- Informações do Projeto ---
    printf("=================================================================================\n");
    printf("  Projeto 1: DGEMM Sequencial e Paralela com OpenMP\n");
    printf("  Disciplina: DEC107 - Processamento Paralelo\n");
    printf("  Autores: Joao Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
    printf("  Data: 29/09/2025\n");
    printf("=================================================================================\n\n");

    // --- Configuração dos Experimentos ---
    int tamanhosMatriz[] = {512, 1024, 2048, 4096}; // Original: matSizes
    int num_tamanhos = sizeof(tamanhosMatriz) / sizeof(int); // Original: num_sizes
    int contagensThreads[] = {2, 4, 8}; // Original: thread_counts
    int num_contagens_threads = sizeof(contagensThreads) / sizeof(int); // Original: num_threads_counts

    printf("Iniciando experimentos de DGEMM...\n");
    printf("Hardware: 4 núcleos físicos, 8 threads\n\n");
    printf("| %-10s | %-12s | %-15s | %-15s | %-15s |\n", "Tamanho", "Threads", "Tempo (s)", "Speedup", "Eficiência");
    printf("|------------|--------------|-----------------|-----------------|-----------------|\n");

    // Loop principal para testar cada tamanho de matriz
    for (int i = 0; i < num_tamanhos; i++) {
        int tamanhoMatriz = tamanhosMatriz[i]; // Original: matSize
        
        // Aloca memória para as matrizes A, B e C
        double *A = alocar_matriz(tamanhoMatriz);
        double *B = alocar_matriz(tamanhoMatriz);
        double *C = alocar_matriz(tamanhoMatriz);

        // Preenche as matrizes A e B com valores aleatórios e zera a matriz C
        inicializar_matrizes(A, B, C, tamanhoMatriz);

        double inicio, fim, tempo_sequencial, tempo_paralelo; // Originais: start, stop, dt_seq, dt_par

        // --- 1. Teste Sequencial (Baseline) ---
        inicio = omp_get_wtime();
        dgemm_sequencial(A, B, C, tamanhoMatriz);
        fim = omp_get_wtime();
        tempo_sequencial = fim - inicio;
        printf("| %-10d | %-12s | %-15.6f | %-15.2f | %-15.2f%% |\n", tamanhoMatriz, "1 (Seq)", tempo_sequencial, 1.0, 100.0);

        // --- 2. Testes Paralelos com OpenMP ---
        for (int j = 0; j < num_contagens_threads; j++) {
            int num_threads = contagensThreads[j]; // Original: n_threads
            omp_set_num_threads(num_threads);
            
            inicializar_matrizes(A, B, C, tamanhoMatriz); // Zera a matriz C para um novo teste

            inicio = omp_get_wtime();
            dgemm_paralelo(A, B, C, tamanhoMatriz);
            fim = omp_get_wtime();
            tempo_paralelo = fim - inicio;

            double aceleracao = tempo_sequencial / tempo_paralelo; // Original: speedup
            double eficiencia = (aceleracao / num_threads) * 100.0; // Original: efficiency
            
            printf("| %-10d | %-12d | %-15.6f | %-15.2f | %-15.2f%% |\n", tamanhoMatriz, num_threads, tempo_paralelo, aceleracao, eficiencia);
        }

        // --- 3. Teste com BLAS (Referência de Alta Performance) ---
        inicializar_matrizes(A, B, C, tamanhoMatriz); // Zera a matriz C
        inicio = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tamanhoMatriz, tamanhoMatriz, tamanhoMatriz,
                    1.0, A, tamanhoMatriz, // alpha = 1.0
                    B, tamanhoMatriz,
                    0.0, C, tamanhoMatriz); // beta = 0.0
        fim = omp_get_wtime();
        double tempo_blas = fim - inicio; // Original: dt_blas
        printf("| %-10d | %-12s | %-15.6f | %-15s | %-15s |\n", tamanhoMatriz, "BLAS", tempo_blas, "-", "-");
        printf("|------------|--------------|-----------------|-----------------|-----------------|\n");

        // FinOps: Liberar a memória assim que não for mais necessária para o tamanho atual.
        // Isso reduz o pico de uso de memória do programa.
        free(A);
        free(B);
        free(C);
    }

    return 0;
}

/**
 * @brief Aloca dinamicamente uma matriz quadrada como um vetor unidimensional.
 * @param dimensao A dimensão (n) da matriz n x n.
 * @return Ponteiro para a área de memória alocada.
 */
double* alocar_matriz(int dimensao) { // Original: matrix_alloc
    // FinOps: Usar (long) para o cálculo do tamanho previne overflow de inteiros
    // para matrizes muito grandes, evitando falhas de alocação e erros.
    double *M = (double*) malloc((long)dimensao * dimensao * sizeof(double));
    if (!M) {
        printf("\nERRO CRÍTICO: Falha na alocacao de memoria.\n");
        exit(-1);
    }
    return M;
}

/**
 * @brief Inicializa as matrizes de entrada A e B com valores aleatórios e zera a matriz C.
 */
void inicializar_matrizes(double *A, double *B, double *C, int dimensao) { // Original: initialize_matrices
    long n_long = dimensao;
    // O loop percorre o vetor linear para inicializar todos os elementos.
    for (long i = 0; i < n_long * n_long; i++) {
        A[i] = (double)(rand() % 3 - 1);
        B[i] = (double)(rand() % 9 - 4);
        C[i] = 0.0;
    }
}

/**
 * @brief Implementação da multiplicação de matrizes de forma sequencial.
 * Referenciada como vSeq.c na especificação.
 * Esta é a versão clássica (ijk) usada como base para medição de speedup.
 */
void dgemm_sequencial(double *A, double *B, double *C, int dimensao) { // Original: dgemm_seq
    long n_long = dimensao;
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double soma = 0.0; // Original: sum
            for (long k = 0; k < n_long; k++) {
                // Acessa A sequencialmente, mas B com saltos, o que pode ser ineficiente para o cache.
                soma += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = soma;
        }
    }
}

/**
 * @brief Implementação da multiplicação de matrizes de forma paralela com OpenMP.
 * FinOps: A ordem dos laços foi alterada para 'ikj' para otimizar o uso da
 * memória cache. Isso reduz a quantidade de cache misses, fazendo com que
 * a CPU espere menos por dados, resultando em maior eficiência e menor tempo
 * de execução.
 */
void dgemm_paralelo(double *A, double *B, double *C, int dimensao) { // Original: dgemm_par
    long n_long = dimensao;
    // A diretiva 'parallel for' distribui as iterações do laço 'i' entre as threads.
    #pragma omp parallel for
    for (long i = 0; i < n_long; i++) {
        for (long k = 0; k < n_long; k++) {
            double a_ik = A[n_long * i + k]; // Reutiliza o valor de A[i][k] no laço interno
            for (long j = 0; j < n_long; j++) {
                // Nesta ordem (ikj), os acessos a B e C são sequenciais,
                // o que é muito mais eficiente para a cache.
                C[n_long * i + j] += a_ik * B[n_long * k + j];
            }
        }
    }
}