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


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>

/* ---------------------------
   Protótipos (funções)
   --------------------------- */
double* matriz_alocar(int n);                       /* matrix_alloc (sequencial) */
void inicializa_matrizes(double *A, double *B, double *C, int n); /* initialize_matrices */
void dgemm_seq_original(double *A, double *B, double *C, int n);  /* FUNÇÃO SEQUENCIAL */
void dgemm_seq_corrigido(double *A, double *B, double *C, int n); /* versão sequencial (sum dentro do loop j) */
void dgemm_par(double *A, double *B, double *C, int n);           /* versão paralela */
void imprimir_cabecalho_projeto(void);
void imprimir_hardware(void);
void zera_matriz(double *M, long n2);

/* ---------------------------
   Função main: programa de experimentos
   --------------------------- */
int main(void) {
    /* Informações do projeto (impresso na saída) */
    imprimir_cabecalho_projeto();
    imprimir_hardware();

    /* Tamanhos e threads para os experimentos */
    int tam_matrizes[] = {512, 1024, 2048, 4096}; /* matSize (original) */
    int num_tam = sizeof(tam_matrizes) / sizeof(int);

    int contagens_threads[] = {2, 4, 8}; /* thread_counts (original) */
    int num_thread_counts = sizeof(contagens_threads) / sizeof(int);

    /* Cabeçalho da tabela de resultados */
    printf("\nResultados dos experimentos (tempos em segundos, GFLOPS, speedup, eficiência):\n");
    printf("| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
           "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiência");
    printf("|----------|----------|--------------|------------|------------|------------|\n");

    /* Seed para rand (não determinístico) */
    srand((unsigned int) time(NULL));

    for (int idx = 0; idx < num_tam; idx++) {
        int tam_matriz = tam_matrizes[idx]; /* matSize (sequencial) */

        /* Alocação de matrizes */
        double *A = matriz_alocar(tam_matriz); /* A (sequencial) */
        double *B = matriz_alocar(tam_matriz); /* B (sequencial) */
        double *C = matriz_alocar(tam_matriz); /* C (sequencial) */

        /* Inicializa as matrizes (A,B rand; C zerada) */
        inicializa_matrizes(A, B, C, tam_matriz);

        double tempo_seq_original = 0.0;
        double tempo_seq_corrigido = 0.0;

        /* ---------------------------
           1) Executa a versão Sequencia
           NOTE: esta função mantém o comportamento do código sequencial.
           --------------------------- */
        zera_matriz(C, (long)tam_matriz * tam_matriz);
        double t0 = omp_get_wtime();
        dgemm_seq_original(A, B, C, tam_matriz); /* função original preservada */
        double t1 = omp_get_wtime();
        tempo_seq_original = t1 - t0;

        /* Compute GFLOPS for original run:
           Número de operações (FLOPs) para multiplicação de matrizes: ~ 2 * n^3
        */
        double flops = 2.0 * (double)tam_matriz * (double)tam_matriz * (double)tam_matriz;
        double gflops_seq_original = (flops / tempo_seq_original) / 1e9;

        /* ---------------------------
           2) Executa a versão Sequencial
           --------------------------- */
        zera_matriz(C, (long)tam_matriz * tam_matriz);
        t0 = omp_get_wtime();
        dgemm_seq_corrigido(A, B, C, tam_matriz);
        t1 = omp_get_wtime();
        tempo_seq_corrigido = t1 - t0;
        double gflops_seq_corrigido = (flops / tempo_seq_corrigido) / 1e9;

        /* Imprime um resumo rápido comparando os dois sequenciais*/
        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "1(orig)", tempo_seq_original, gflops_seq_original, "-", "-");

        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "1(corr)", tempo_seq_corrigido, gflops_seq_corrigido, "-", "-");

        /* Para os testes paralelos usaremos o tempo como baseline (dt_seq) */
        double dt_seq = tempo_seq_corrigido;

        /* ---------------------------
           3) Executa versões paralelas para diferentes números de threads
           --------------------------- */
        for (int t = 0; t < num_thread_counts; t++) {
            int n_threads = contagens_threads[t];
            omp_set_num_threads(n_threads);

            /* reinicializa C para zero antes do teste paralelo (condições idênticas) */
            inicializa_matrizes(A, B, C, tam_matriz); /* inicializa coloca C em zero também */
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
           4) Teste com BLAS (cblas_dgemm)
           --------------------------- */
        inicializa_matrizes(A, B, C, tam_matriz); /* C = 0 */
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

        /* libera memória */
        free(A);
        free(B);
        free(C);
    }

    /* Final */
    printf("\nExperimentos finalizados. Verifique os dados acima para compor o relatório PDF.\n");
    return 0;
}

/* ---------------------------
   Implementações das funções
   --------------------------- */

/* matriz_alocar: mesma ideia de matrix_alloc (sequencial) */
double* matriz_alocar(int n) {
    /* matrix_alloc (sequencial) */
    /* Usamos (size_t) para evitar overflow em multiplos sistemas */
    double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
    if (!M) {
        printf("\nErro: Alocacao de memoria\n");
        exit(-1);
    }
    return M;
}

/* inicializa_matrizes: popula A,B com valores pseudo-aleatórios e zera C
   corresponde a initialize_matrices (sequencial) */
void inicializa_matrizes(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    long total = n_long * n_long;
    for (long i = 0; i < total; i++) {
        A[i] = (double)(rand() % 3 - 1);  /* mesmo padrão do código sequencial */
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

/* dgemm_seq_corrigido: versão sequencial(sum reiniciado por j)
   corresponde à dgemm_seq(com sum local por j)
*/
void dgemm_seq_corrigido(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double sum = 0.0; /* sum privado por (i,j) */
            for (long k = 0; k < n_long; k++) {
                sum += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = sum;
        }
    }
}

/* dgemm_par: versão paralela usando OpenMP (corresponde a dgemm_par) */
void dgemm_par(double *A, double *B, double *C, int n) {
    long n_long = (long) n;
    #pragma omp parallel for
    for (long i = 0; i < n_long; i++) {
        for (long j = 0; j < n_long; j++) {
            double sum = 0.0; /* sum é privada para cada iteração do loop j */
            for (long k = 0; k < n_long; k++) {
                sum += A[n_long * i + k] * B[n_long * k + j];
            }
            C[n_long * i + j] = sum;
        }
    }
}

/* zera_matriz: zera um vetor de tamanho n2 (útil para reinicializar C) */
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
    printf("\nHardware usado (informado no enunciado):\n");
    printf("- Notebook Dell Inspiron 15 3520\n");
    printf("- Sistema Operacional: Ubuntu 24.04.2 LTS (via WSL2) (conforme enunciado)\n");
    printf("- Processador: Intel Core i5-1135G7 (4 núcleos físicos, 8 threads) ~ 2.433 GHz\n");
    printf("- Memória RAM: 18 GB DDR4 (informado)\n");
    printf("- Cache L2: 5 MB; Cache L3: 8 MB (informado)\n");
    printf("- Compilador sugerido: GCC 13.3.0 com flags -O3 -fopenmp -march=native (opcional)\n");
    printf("\n");
}

/* FIM DO ARQUIVO */
