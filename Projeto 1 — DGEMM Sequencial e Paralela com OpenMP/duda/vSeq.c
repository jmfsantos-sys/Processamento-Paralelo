// Compilar sem a BLAS e com openMP: gcc -O3 -fopenmp -o vSeq vSeq.c
// Compilar com a BLAS: gcc -O3 -fopenmp -o vSeq vSeq.c -lblas -lopenblas

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>

double* matrix_alloc(int n);
void dgemm_seq(double *A, double *B, double *C, int n);

int main(void) {
    // Testes
    // int matSize = 2;
    int matSize = 512;
    // int matSize = 1024;
    // int matSize = 2048; 
    // int matSize = 4096;

    // Gera as matrizes
    double *A = matrix_alloc(matSize);
    double *B = matrix_alloc(matSize);
    double *C = matrix_alloc(matSize);

    // Inicializa valores nas matrizes
    for (int i=0; i<matSize; i++) {
        for (int j=0; j<matSize; j++) {
            A[matSize*i + j] = (double)(rand()%3 - 1);
            // printf("\nA[%d][%d]= %lf\t", i, j, A[matSize*i + j]);
            B[matSize*i + j] = (double)(rand()%9 - 4);
            // printf("\nB[%d][%d]= %lf\t", i, j, B[matSize*i + j]);
            C[matSize*i + j] = 0.0;
            // printf("\nC[%d][%d]= %lf\t", i, j, C[matSize*i + j]);
        }
        // printf("\n");
    }

    double start, stop, dt;

    // Realiza a multiplicação das matrizes e tempo de execução
    start = omp_get_wtime();
    dgemm_seq(A, B, C, matSize);
    stop = omp_get_wtime();
    dt = stop - start;

    printf("Tempo de execucao do dgemm: %.10lf segundos\n", dt);

    // Compara com a função da BLAS (cblas_dgemm)

    // Zerando a matriz C
    for (int i=0; i<matSize*matSize; i++) 
        C[i] = 0.0; 

    start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,    
                matSize, matSize, matSize,
                1.0, A, matSize, // alpha=1.0
                B, matSize,
                0.0, C, matSize); // beta=0.0      
    stop = omp_get_wtime();
    dt = stop - start;

    printf("Tempo de execucao da BLAS: %.10lf segundos\n", dt);

    free(A);
    free(B);
    free(C);
    return 0;
}

double* matrix_alloc(int n) {
    double *M = (double*) malloc(n*n*sizeof(double));
    if (!M) {
        printf("\nErro: Alocacao de memoria");
        exit(-1);
    }

    return M;
}

void dgemm_seq(double *A, double *B, double *C, int n) {
    double sum = 0.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            for (int k=0; k<n; k++) {
                sum += A[n*i + k] * B[n*k + j];
            }

            C[n*i + j] = sum;
        }
    }
}