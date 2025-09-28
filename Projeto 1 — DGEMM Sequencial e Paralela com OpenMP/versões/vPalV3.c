/*
 ============================================================================
 Arquivo: vPalV2.c
 Projeto 1 — Multiplicação de Matrizes (DGEMM) Sequencial e Paralela com OpenMP
 ============================================================================
 Disciplina: DEC107 — Processamento Paralelo
 Curso: Bacharelado em Ciência da Computação
 Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
 Data: 29/09/2025

 Instruções de compilação:
  - Sem BLAS, com OpenMP:
      gcc -O3 -fopenmp -o vPalV2 vPalV2.c
  - Com BLAS (ex.: OpenBLAS) e OpenMP:
      gcc -O3 -fopenmp -o vPalV2 vPalV2.c -lopenblas
  - Observação: para usar cblas_dgemm (BLAS) é necessário ter libblas / libopenblas instalada
      (link com -lopenblas ou -lblas dependendo do seu sistema).

 Objetivo:
  - Implementar e comparar versões sequencial e paralela (OpenMP) de DGEMM
  - Validar com BLAS (cblas_dgemm) e medir tempo, speedup, eficiência e GFLOPS.
 ============================================================================
*/

/* Bibliotecas principais */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>

/* ---------------------------
   Protótipos de funções
   --------------------------- */
double* matriz_alocar(int n);                       
void inicializa_matrizes(double *A, double *B, double *C, int n);
void dgemm_seq_original(double *A, double *B, double *C, int n);  
void dgemm_seq_corrigido(double *A, double *B, double *C, int n); 
void dgemm_par(double *A, double *B, double *C, int n);           
void imprimir_cabecalho_projeto(void);
void imprimir_hardware(void);
void zera_matriz(double *M, long n2);

/* ---------------------------
   Função main: programa de experimentos
   --------------------------- */
int main(void) {

    /* =======================================================
       Cabeçalho do projeto e informações do hardware
       ======================================================= */
    imprimir_cabecalho_projeto();
    imprimir_hardware();

    /* =======================================================
       Requisitos do professor ==
       ======================================================= */
    printf("\n=== Requisitos do projeto ===\n");
    printf("1) Versão sequencial implementada (dgemm_seq_corrigido)\n");
    printf("2) Versão paralela implementada com OpenMP (dgemm_par)\n");
    printf("3) Medição de tempo com omp_get_wtime()\n");
    printf("4) Testes realizados para matrizes 512, 1024, 2048, 4096\n");
    printf("5) Testes paralelos com 2, 4 e 8 threads\n");
    printf("6) Cálculo de GFLOPS, speedup e eficiência\n");
    printf("7) Validação opcional com BLAS (cblas_dgemm)\n");
    printf("========================================================\n");

    /* =======================================================
       Configurações de experimentos
       ======================================================= */
    int tam_matrizes[] = {512, 1024, 2048, 4096};
    int num_tam = sizeof(tam_matrizes) / sizeof(int);
    int contagens_threads[] = {2, 4, 8};
    int num_thread_counts = sizeof(contagens_threads) / sizeof(int);

    /* Cabeçalho da tabela de resultados */
    printf("\nResultados dos experimentos (tempos em segundos, GFLOPS, speedup, eficiência):\n");
    printf("| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
           "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiência");
    printf("|----------|----------|--------------|------------|------------|------------|\n");

    /* Seed para rand() */
    srand((unsigned int) time(NULL));

    for (int idx = 0; idx < num_tam; idx++) {
        int tam_matriz = tam_matrizes[idx];

        /* Alocação das matrizes */
        double *A = matriz_alocar(tam_matriz);
        double *B = matriz_alocar(tam_matriz);
        double *C = matriz_alocar(tam_matriz);

        /* Inicializa matrizes A e B com valores aleatórios, C zerada */
        inicializa_matrizes(A, B, C, tam_matriz);

        double tempo_seq_original = 0.0;
        double tempo_seq_corrigido = 0.0;

        /* ---------------------------
           1) Executa a versão sequencial
           --------------------------- */
        zera_matriz(C, (long)tam_matriz * tam_matriz);
        double t0 = omp_get_wtime();
        dgemm_seq_original(A, B, C, tam_matriz);
        double t1 = omp_get_wtime();
        tempo_seq_original = t1 - t0;
        double flops = 2.0 * (double)tam_matriz * (double)tam_matriz * (double)tam_matriz;
        double gflops_seq_original = (flops / tempo_seq_original) / 1e9;

        /* ---------------------------
           2) Executa a versão sequencial (fdls)
           --------------------------- */
        zera_matriz(C, (long)tam_matriz * tam_matriz);
        t0 = omp_get_wtime();
        dgemm_seq_corrigido(A, B, C, tam_matriz);
        t1 = omp_get_wtime();
        tempo_seq_corrigido = t1 - t0;
        double gflops_seq_corrigido = (flops / tempo_seq_corrigido) / 1e9;

        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "1(orig)", tempo_seq_original, gflops_seq_original, "-", "-");
        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "1(corr)", tempo_seq_corrigido, gflops_seq_corrigido, "-", "-");

        /* Tempo baseline para cálculo de speedup */
        double dt_seq = tempo_seq_corrigido;

        /* ---------------------------
           3) Testes paralelos com diferentes números de threads
           --------------------------- */
        for (int t = 0; t < num_thread_counts; t++) {
            int n_threads = contagens_threads[t];
            omp_set_num_threads(n_threads);

            /* Reinicializa A/B/C para condições idênticas */
            inicializa_matrizes(A, B, C, tam_matriz);
            zera_matriz(C, (long)tam_matriz * tam_matriz);

            double tstart = omp_get_wtime();
            dgemm_par(A, B, C, tam_matriz);
            double tstop = omp_get_wtime();
            double dt_par = tstop - tstart;

            double speedup = dt_seq / dt_par;
            double eficiencia = (speedup / (double)n_threads) * 100.0;
            double gflops_par = (flops / dt_par) / 1e9;

            printf("| %-8d | %-8d | %-12.6f | %-10.3f | %-10.3f | %-9.2f%% |\n",
                   tam_matriz, n_threads, dt_par, gflops_par, speedup, eficiencia);
        }

        /* ---------------------------
           4) Teste com BLAS
           --------------------------- */
        inicializa_matrizes(A, B, C, tam_matriz);
        double tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tam_matriz, tam_matriz, tam_matriz,
                    1.0, A, tam_matriz,
                    B, tam_matriz,
                    0.0, C, tam_matriz);
        double tstop = omp_get_wtime();
        double dt_blas = tstop - tstart;
        double gflops_blas = (flops / dt_blas) / 1e9;

        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "BLAS", dt_blas, gflops_blas, "-", "-");

        printf("|----------|----------|--------------|------------|------------|------------|\n");

        /* Libera memória */
        free(A);
        free(B);
        free(C);
    }

    printf("\nExperimentos finalizados. Verifique os dados acima para compor o relatório PDF.\n");
    return 0;
}

/* ---------------------------
   Funções utilitárias
   --------------------------- */

double* matriz_alocar(int n) {
    double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
    if (!M) {
        printf("\nErro: Alocacao de memoria\n");
        exit(-1);
    }
    return M;
}

void inicializa_matrizes(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    long total = n_long * n_long;
    for (long i = 0; i < total; i++) {
        A[i] = (double)(rand() % 3 - 1);
        B[i] = (double)(rand() % 9 - 4);
        C[i] = 0.0;
    }
}

void dgemm_seq_original(double *A, double *B, double *C, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                sum += A[n*i + k] * B[n*k + j];
            }
            C[n*i + j] = sum;
        }
    }
}

void dgemm_seq_corrigido(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double sum = 0.0;
            for (long k = 0; k < n_long; k++) {
                sum += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = sum;
        }
    }
}

void dgemm_par(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    #pragma omp parallel for
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double sum = 0.0;
            for (long k = 0; k < n_long; k++) {
                sum += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = sum;
        }
    }
}

void zera_matriz(double *M, long n2) {
    for (long i = 0; i < n2; i++) M[i] = 0.0;
}

void imprimir_cabecalho_projeto(void) {
    printf("=============================================================\n");
    printf("Projeto 1 — DGEMM Sequencial e Paralela com OpenMP\n");
    printf("Disciplina: DEC107 — Processamento Paralelo\n");
    printf("Autores (dupla): João Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
    printf("Data de entrega: 29/09/2025\n");
    printf("Objetivo: Implementar versões sequencial e paralela de DGEMM, comparar com BLAS\n");
    printf("Entregáveis: códigos-fonte e relatório de desempenho (PDF)\n");
    printf("Observação: o código original do aluno foi preservado na função 'dgemm_seq_original'.\n");
    printf("=============================================================\n");
}

void imprimir_hardware(void) {
    printf("\nHardware usado nos testes:\n");
    printf("- Notebook Dell Inspiron 15 3520\n");
    printf("- Sistema Operacional: Windows 11 Home SL, versão 24H2\n");
    printf("- CPU: Intel Core i5-1135G7 (4 núcleos, 8 threads, 2.40 GHz)\n");
    printf("- Memória RAM: 16 GB DDR4 3200 MT/s\n");
    printf("- Armazenamento: SSD NVMe 256 GB (SK Hynix BC711)\n");
    printf("- GPU: Intel Iris Xe (integrada)\n");
    printf("- Tela: 1920x1080 Full HD\n");
    printf("- BIOS: Dell Inc. 1.34.0 (16/07/2024)\n");
    printf("- Rede Wi-Fi: Realtek 8821CE; Bluetooth: Realtek\n\n");
}

/* FIM DO ARQUIVO */
