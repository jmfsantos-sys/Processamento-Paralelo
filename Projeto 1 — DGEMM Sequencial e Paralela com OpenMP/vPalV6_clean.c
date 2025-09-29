/*
=============================================================================
Projeto 1  Multiplicao de Matrizes (DGEMM) Sequencial e Paralela com OpenMP
=============================================================================
Disciplina: DEC107  Processamento Paralelo
Curso: Bacharelado em Cincia da Computao
Autores: Joao Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Data: 27/09/2025

Objetivo do projeto:
 - Implementar e comparar verses sequencial e paralela de DGEMM.
 - Medir tempo, GFLOPS, speedup e eficincia real.
 - Comparar com referncia de alta performance (BLAS).

=============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>

void imprimir_informacoes_iniciais(void);
void imprimir_hardware(void);
double* matriz_alocar(int n);
void inicializa_matrizes(double *A, double *B, double *C, int n);
void dgemm_sequencial_otimizado(double *A, double *B, double *C, int n);
void dgemm_paralelo_otimizado(double *A, double *B, double *C, int n);

int main(void) {
  imprimir_informacoes_iniciais();

  int tam_matrizes[] = {512, 1024, 2048, 4096};
  int num_tam = sizeof(tam_matrizes) / sizeof(int);
  int contagens_threads[] = {2, 4, 8};
  int num_thread_counts = sizeof(contagens_threads) / sizeof(int);
  const int NUM_REPETICOES = 3; // Mdia de 3 execues

  printf("\n--- INCIO DOS EXPERIMENTOS (Mdia de %d execues) ---\n", NUM_REPETICOES);
  printf("| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
     "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficincia");
  printf("|----------|----------|--------------|------------|------------|------------|\n");

  srand((unsigned int)time(NULL));

  for (int idx = 0; idx < num_tam; idx++) {
    int tam_matriz = tam_matrizes[idx];
    long n_long = tam_matriz;

    double *A = matriz_alocar(tam_matriz);
    double *B = matriz_alocar(tam_matriz);
    double *C = matriz_alocar(tam_matriz);

    double flops = 2.0 * n_long * n_long * n_long;

    printf("\n[LOG] Executando testes para matriz de tamanho %dx%d...\n", tam_matriz, tam_matriz);

    double tempo_seq_total = 0.0;
    for (int rep = 0; rep < NUM_REPETICOES; rep++) {
      inicializa_matrizes(A, B, C, tam_matriz);
      double t0 = omp_get_wtime();
      dgemm_sequencial_otimizado(A, B, C, tam_matriz);
      double t1 = omp_get_wtime();
      tempo_seq_total += (t1 - t0);
    }
    double tempo_seq_medio = tempo_seq_total / NUM_REPETICOES;
    double gflops_seq = (flops / tempo_seq_medio) / 1e9;
    printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
       tam_matriz, "1 (Seq)", tempo_seq_medio, gflops_seq, "1.00x", "100.00%");

    for (int t = 0; t < num_thread_counts; t++) {
      int n_threads = contagens_threads[t];
      omp_set_num_threads(n_threads);

      double tempo_par_total = 0.0;
      for (int rep = 0; rep < NUM_REPETICOES; rep++) {
        inicializa_matrizes(A, B, C, tam_matriz);
        double tstart = omp_get_wtime();
        dgemm_paralelo_otimizado(A, B, C, tam_matriz);
        double tstop = omp_get_wtime();
        tempo_par_total += (tstop - tstart);
      }
      double tempo_par_medio = tempo_par_total / NUM_REPETICOES;

      double speedup = tempo_seq_medio / tempo_par_medio;
      double eficiencia = (speedup / (double)n_threads) * 100.0; 
      double gflops_par = (flops / tempo_par_medio) / 1e9;
      printf("| %-8d | %-8d | %-12.6f | %-10.3f | %-10.3fx | %-9.2f%% |\n",
         tam_matriz, n_threads, tempo_par_medio, gflops_par, speedup, eficiencia);
    }

    double tempo_blas_total = 0.0;
    for (int rep = 0; rep < NUM_REPETICOES; rep++) {
      inicializa_matrizes(A, B, C, tam_matriz);
      double tstart_blas = omp_get_wtime();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            tam_matriz, tam_matriz, tam_matriz,
            1.0, A, tam_matriz, B, tam_matriz, 0.0, C, tam_matriz);
      double tstop_blas = omp_get_wtime();
      tempo_blas_total += (tstop_blas - tstart_blas);
    }
    double tempo_blas_medio = tempo_blas_total / NUM_REPETICOES;
    double gflops_blas = (flops / tempo_blas_medio) / 1e9;
    printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
       tam_matriz, "BLAS", tempo_blas_medio, gflops_blas, "-", "-");

    printf("|----------|----------|--------------|------------|------------|------------|\n");

    free(A); free(B); free(C);
  }

  printf("\n[LOG] Experimentos finalizados.\n");
  return 0;
}

void dgemm_sequencial_otimizado(double *A, double *B, double *C, int n) {
  long n_long = n;
  for (long i = 0; i < n_long; i++) {
    for (long k = 0; k < n_long; k++) {
      double a_ik = A[n_long * i + k];
      for (long j = 0; j < n_long; j++) {
        C[n_long * i + j] += a_ik * B[n_long * k + j];
      }
    }
  }
}

void dgemm_paralelo_otimizado(double *A, double *B, double *C, int n) {
  long n_long = n;
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

double* matriz_alocar(int n) {
  double *M = (double*) malloc((size_t)n * (size_t)n * sizeof(double));
  if (!M) { printf("\nERRO: Falha na alocacao de memoria.\n"); exit(-1); }
  return M;
}

void inicializa_matrizes(double *A, double *B, double *C, int n) {
  long total = (long)n * n;
  #pragma omp parallel for
  for (long i = 0; i < total; i++) {
    unsigned int seed = omp_get_thread_num() + (unsigned int)time(NULL);
    A[i] = (double)(rand_r(&seed) % 3 - 1);
    B[i] = (double)(rand_r(&seed) % 9 - 4);
    C[i] = 0.0;
  }
}

void imprimir_informacoes_iniciais(void) {
  printf("=============================================================\n");
  printf("   Projeto 1  DGEMM Sequencial e Paralela com OpenMP\n");
  printf("=============================================================\n");
  printf("Autores: Joo Manoel Fidelis Santos e Maria Eduarda Guedes Alves\n");
  printf("Disciplina: DEC107  Processamento Paralelo\n\n");
  printf("Hardware dos computadores utilizados:\n");
  imprimir_hardware();
}

void imprimir_hardware(void) {
  printf("\n--- Hardware de Maria Eduarda ---\n");
  printf(" - CPU: Intel Core i7-10750H (6 ncleos, 12 threads) @ 2.6GHz\n");
  printf(" - RAM: 16 GB DDR4\n");
  printf(" - GPU: NVIDIA GTX 1660 Ti\n");
  printf(" - SSD: 512 GB NVMe\n");
  printf(" - OS: Windows 10 Home\n");
  printf("----------------------------------\n");
  printf("\n--- Hardware de Joao Manoel ---\n");
  printf(" - CPU: AMD Ryzen 5 5600G (6 ncleos, 12 threads)\n");
  printf(" - Placa-me: SOYO SY-Classic B450M\n");
  printf(" - RAM: 40 GB DDR4 (Dual Channel)\n");
  printf(" - GPU: AMD Radeon Graphics (Integrada, Vega)\n");
  printf("----------------------------------\n");

}
