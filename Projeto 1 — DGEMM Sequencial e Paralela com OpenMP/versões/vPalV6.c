#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>

// --- Protótipos das Funções ---
double* matriz_alocar(int n);
void inicializa_matrizes(double *A, double *B, double *C, int n);
void dgemm_sequencial(double *A, double *B, double *C, int n);
void dgemm_paralelo(double *A, double *B, double *C, int n);
void imprimir_cabecalho_projeto(void);
void imprimir_requisitos(void);
void selecionar_hardware(void);

// --- Função Principal ---
int main(void) {

    // Pergunta de quem é o computador e exibe as informações correspondentes
    selecionar_hardware();

    imprimir_cabecalho_projeto();
    imprimir_requisitos();

    int tam_matrizes[] = {512, 1024, 2048, 4096};
    int num_tam = sizeof(tam_matrizes) / sizeof(int);

    int contagens_threads[] = {2, 4, 8};
    int num_thread_counts = sizeof(contagens_threads) / sizeof(int);

    printf("\n--- INÍCIO DOS EXPERIMENTOS ---\n");
    printf("| %-8s | %-8s | %-12s | %-10s | %-10s | %-10s |\n",
           "Tamanho", "Threads", "Tempo (s)", "GFLOPS", "Speedup", "Eficiência");
    printf("|----------|----------|--------------|------------|------------|------------|\n");

    srand((unsigned int) time(NULL));

    for (int idx = 0; idx < num_tam; idx++) {
        int tam_matriz = tam_matrizes[idx];
        long n_long = tam_matriz;
        long n2 = n_long * n_long;

        double *A = matriz_alocar(tam_matriz);
        double *B = matriz_alocar(tam_matriz);
        double *C = matriz_alocar(tam_matriz);

        double tempo_seq;
        double flops = 2.0 * n_long * n_long * n_long;

        // --- 1. Teste Sequencial ---
        printf("\n[LOG] Executando teste para matriz de tamanho %dx%d...\n", tam_matriz, tam_matriz);
        inicializa_matrizes(A, B, C, tam_matriz);

        double t0 = omp_get_wtime();
        dgemm_sequencial(A, B, C, tam_matriz);
        double t1 = omp_get_wtime();
        tempo_seq = t1 - t0;

        double gflops_seq = (flops / tempo_seq) / 1e9;
        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "1 (Seq)", tempo_seq, gflops_seq, "1.00x", "100.00%");

        // --- 2. Testes Paralelos ---
        for (int t = 0; t < num_thread_counts; t++) {
            int n_threads = contagens_threads[t];
            omp_set_num_threads(n_threads);

            inicializa_matrizes(A, B, C, tam_matriz);

            double tstart = omp_get_wtime();
            dgemm_paralelo(A, B, C, tam_matriz);
            double tstop = omp_get_wtime();
            double tempo_par = tstop - tstart;

            double speedup = tempo_seq / tempo_par;
            double eficiencia = (speedup / (double)n_threads) * 100.0;
            double gflops_par = (flops / tempo_par) / 1e9;

            printf("| %-8d | %-8d | %-12.6f | %-10.3f | %-10.3fx | %-9.2f%% |\n",
                   tam_matriz, n_threads, tempo_par, gflops_par, speedup, eficiencia);
        }

        // --- 3. Teste com BLAS ---
        inicializa_matrizes(A, B, C, tam_matriz);

        double tstart_blas = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tam_matriz, tam_matriz, tam_matriz,
                    1.0, A, tam_matriz, B, tam_matriz, 0.0, C, tam_matriz);
        double tstop_blas = omp_get_wtime();

        double tempo_blas = tstop_blas - tstart_blas;
        double gflops_blas = (flops / tempo_blas) / 1e9;
        printf("| %-8d | %-8s | %-12.6f | %-10.3f | %-10s | %-10s |\n",
               tam_matriz, "BLAS", tempo_blas, gflops_blas, "-", "-");

        printf("|----------|----------|--------------|------------|------------|------------|\n");

        free(A);
        free(B);
        free(C);
    }

    printf("\n[LOG] Experimentos finalizados.\n");
    return 0;
}

// --- Função para selecionar hardware ---
void selecionar_hardware(void) {
    int opcao = 0;
    printf("De quem é o computador?\n");
    printf("1 - Maria Eduarda\n");
    printf("2 - João Manoel\n");
    printf("Escolha (1 ou 2): ");
    scanf("%d", &opcao);

    printf("\n=== Informações do Computador ===\n");
    if (opcao == 1) {
        printf("Hardware do computador de Maria Eduarda (original):\n");
        printf("CPU: Intel Core i7-10750H, 6 núcleos, 12 threads @ 2.6GHz\n");
        printf("RAM: 16 GB DDR4\n");
        printf("GPU: NVIDIA GTX 1660 Ti\n");
        printf("SSD: 512 GB NVMe\n");
        printf("OS: Windows 10 Home\n");
    } else if (opcao == 2) {
        printf("Hardware do computador de João Manoel:\n");
        printf("CPU: Intel Core i5-1135G7, 4 núcleos, 8 threads @ 2.40GHz\n");
        printf("RAM: 16 GB DDR4 3200 MHz\n");
        printf("GPU: Intel Iris Xe Graphics (integrada)\n");
        printf("SSD: SK Hynix BC711 NVMe 256 GB\n");
        printf("OS: Windows 11 Home Single Language\n");
    } else {
        printf("Opção inválida! Exibindo informações padrão.\n");
    }
    printf("================================\n\n");
}
