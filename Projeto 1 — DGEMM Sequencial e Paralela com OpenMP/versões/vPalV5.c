/*****************************************************************************************
 * *
 * Título: Projeto 1 — DGEMM Sequencial e Paralela com OpenMP                             *
 * Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves                      *
 * Disciplina: DEC107 — Processamento Paralelo                                           *
 * Data: 29/09/2025                                                                      *
 * Arquivo: vPalV2_com_selecao.c                                                         *
 * *
 * Objetivo do Projeto:                                                                  *
 * Implementar e avaliar o desempenho de rotinas de multiplicação de matrizes            *
 * (DGEMM) em versões sequencial e paralela (com OpenMP).                                *
 * *
 * Como Compilar:                                                                        *
 * gcc -O3 -fopenmp -o dgemm_teste vPalV2_com_selecao.c -lopenblas                        *
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
void imprimir_hardware_maria(void);
void imprimir_hardware_joao(void);

// --- Função Principal ---
int main(void) {
    // --- Informações do Projeto ---
    printf("=================================================================================\n");
    printf("  Projeto 1: DGEMM Sequencial e Paralela com OpenMP\n");
    printf("  Disciplina: DEC107 - Processamento Paralelo\n");
    printf("  Autores: Joao Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
    printf("  Data: 29/09/2025\n");
    printf("=================================================================================\n\n");

    // --- SELEÇÃO DE HARDWARE ---
    int escolha_pc = 0;
    printf("Selecione o computador onde os testes serao executados:\n");
    printf("  1 - Computador de Maria Eduarda\n");
    printf("  2 - Computador de Joao Manoel\n");
    printf("Escolha (1 ou 2): ");
    scanf("%d", &escolha_pc);

    if (escolha_pc == 1) {
        imprimir_hardware_maria();
    } else if (escolha_pc == 2) {
        imprimir_hardware_joao();
    } else {
        printf("\nOpcao invalida. Encerrando o programa.\n");
        return 1;
    }

    // --- Configuração dos Experimentos ---
    int tamanhosMatriz[] = {512, 1024, 2048, 4096};
    int num_tamanhos = sizeof(tamanhosMatriz) / sizeof(int);
    int contagensThreads[] = {2, 4, 8};
    int num_contagens_threads = sizeof(contagensThreads) / sizeof(int);

    printf("Iniciando experimentos de DGEMM...\n\n");
    printf("| %-10s | %-12s | %-15s | %-15s | %-15s |\n", "Tamanho", "Threads", "Tempo (s)", "Speedup", "Eficiência");
    printf("|------------|--------------|-----------------|-----------------|-----------------|\n");

    for (int i = 0; i < num_tamanhos; i++) {
        int tamanhoMatriz = tamanhosMatriz[i];
        
        double *A = alocar_matriz(tamanhoMatriz);
        double *B = alocar_matriz(tamanhoMatriz);
        double *C = alocar_matriz(tamanhoMatriz);

        inicializar_matrizes(A, B, C, tamanhoMatriz);

        double inicio, fim, tempo_sequencial, tempo_paralelo;

        // --- 1. Teste Sequencial (Baseline) ---
        inicio = omp_get_wtime();
        dgemm_sequencial(A, B, C, tamanhoMatriz);
        fim = omp_get_wtime();
        tempo_sequencial = fim - inicio;
        printf("| %-10d | %-12s | %-15.6f | %-15.2f | %-15.2f%% |\n", tamanhoMatriz, "1 (Seq)", tempo_sequencial, 1.0, 100.0);

        // --- 2. Testes Paralelos com OpenMP ---
        for (int j = 0; j < num_contagens_threads; j++) {
            int num_threads = contagensThreads[j];
            omp_set_num_threads(num_threads);
            
            inicializar_matrizes(A, B, C, tamanhoMatriz);

            inicio = omp_get_wtime();
            dgemm_paralelo(A, B, C, tamanhoMatriz);
            fim = omp_get_wtime();
            tempo_paralelo = fim - inicio;

            double aceleracao = tempo_sequencial / tempo_paralelo;
            double eficiencia = (aceleracao / num_threads) * 100.0;
            
            printf("| %-10d | %-12d | %-15.6f | %-15.2f | %-15.2f%% |\n", tamanhoMatriz, num_threads, tempo_paralelo, aceleracao, eficiencia);
        }

        // --- 3. Teste com BLAS (Referência) ---
        inicializar_matrizes(A, B, C, tamanhoMatriz);
        inicio = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tamanhoMatriz, tamanhoMatriz, tamanhoMatriz,
                    1.0, A, tamanhoMatriz,
                    B, tamanhoMatriz,
                    0.0, C, tamanhoMatriz);
        fim = omp_get_wtime();
        double tempo_blas = fim - inicio;
        printf("| %-10d | %-12s | %-15.6f | %-15s | %-15s |\n", tamanhoMatriz, "BLAS", tempo_blas, "-", "-");
        printf("|------------|--------------|-----------------|-----------------|-----------------|\n");

        free(A);
        free(B);
        free(C);
    }

    return 0;
}

double* alocar_matriz(int dimensao) {
    double *M = (double*) malloc((long)dimensao * dimensao * sizeof(double));
    if (!M) {
        printf("\nERRO CRÍTICO: Falha na alocacao de memoria.\n");
        exit(-1);
    }
    return M;
}

void inicializar_matrizes(double *A, double *B, double *C, int dimensao) {
    long n_long = dimensao;
    for (long i = 0; i < n_long * n_long; i++) {
        A[i] = (double)(rand() % 3 - 1);
        B[i] = (double)(rand() % 9 - 4);
        C[i] = 0.0;
    }
}

void dgemm_sequencial(double *A, double *B, double *C, int dimensao) {
    long n_long = dimensao;
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double soma = 0.0;
            for (long k = 0; k < n_long; k++) {
                soma += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = soma;
        }
    }
}

void dgemm_paralelo(double *A, double *B, double *C, int dimensao) {
    long n_long = dimensao;
    #pragma omp parallel for
    for (long i = 0; i < n_long; i++) {
        for (long k = 0; k < n_long; k++) {
            double a_ik = A[n_long * i + k];
            for (long j = 0; j < n_long; j++) {
                C[n_long * i + j] += a_ik * B[n_long * k + j];
            }
        }
    }
}

void imprimir_hardware_maria(void) {
    printf("\n--- Hardware de Maria Eduarda ---\n");
    printf("  - Modelo:           Notebook Dell Inspiron 15 3520\n");
    printf("  - Processador (CPU): Intel Core i5-1135G7 (4 Nucleos, 8 Threads) @ 2.40GHz\n");
    printf("  - Memoria RAM:      16 GB DDR4 @ 3200 MHz\n");
    printf("  - Armazenamento:    SSD NVMe 256 GB SK Hynix BC711\n");
    printf("  - Placa de Video:   Intel Iris Xe Graphics (Integrada)\n");
    printf("  - Sistema Op.:      Microsoft Windows 11 Home Single Language\n\n");
}

void imprimir_hardware_joao(void) {
    printf("\n--- Hardware de Joao Manoel ---\n");
    printf("  - Processador (CPU): Intel Core i5-1135G7 (4 Nucleos, 8 Threads) @ 2.40GHz\n");
    printf("  - Memoria RAM:      16 GB DDR4 @ 3200 MHz\n");
    printf("  - Placa de Video:   Intel Iris Xe Graphics (Integrada)\n");
    printf("  - Armazenamento:    SSD NVMe 256 GB SK Hynix BC711\n");
    printf("  - Sistema Op.:      Microsoft Windows 11 Home Single Language\n\n");
}