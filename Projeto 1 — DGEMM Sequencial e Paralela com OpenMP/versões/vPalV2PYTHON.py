"""
Projeto 1 — DGEMM Sequencial e Paralela com Multiprocessing
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Disciplina: DEC107 — Processamento Paralelo
Data: 29/09/2025
"""

import numpy as np
import time
import multiprocessing as mp
import argparse

# ----------------------
# Funções de inicialização
# ----------------------
def inicializar_matrizes(dimensao):
    """Cria e inicializa matrizes A, B e C."""
    A = np.random.randint(-1, 2, size=(dimensao, dimensao), dtype=np.float64)
    B = np.random.randint(-4, 5, size=(dimensao, dimensao), dtype=np.float64)
    C = np.zeros((dimensao, dimensao), dtype=np.float64)
    return A, B, C

# ----------------------
# Multiplicação sequencial
# ----------------------
def dgemm_sequencial(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma += A[i, k] * B[k, j]
            C[i, j] = soma
    return C

# ----------------------
# Multiplicação paralela com multiprocessing
# ----------------------
def worker_multiplicar(args):
    """Worker para multiplicação de um bloco de linhas de A."""
    A_block, B = args
    n_rows = A_block.shape[0]
    n_cols = B.shape[1]
    C_block = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            soma = 0.0
            for k in range(B.shape[0]):
                soma += A_block[i, k] * B[k, j]
            C_block[i, j] = soma
    return C_block

def dgemm_paralelo(A, B, num_threads):
    """Divide A em blocos de linhas e multiplica em paralelo."""
    n = A.shape[0]
    # Determina blocos para cada processo
    block_sizes = [(i*n//num_threads, (i+1)*n//num_threads) for i in range(num_threads)]
    args = [(A[start:end, :], B) for start, end in block_sizes]

    with mp.Pool(processes=num_threads) as pool:
        results = pool.map(worker_multiplicar, args)

    # Concatena blocos
    C = np.vstack(results)
    return C

# ----------------------
# Script principal
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGEMM Sequencial e Paralela com Multiprocessing")
    parser.add_argument("--threads", type=int, nargs='*', default=[2, 4, 8],
                        help="Número de threads para paralelismo")
    parser.add_argument("--sizes", type=int, nargs='*', default=[512, 1024, 2048],
                        help="Tamanhos das matrizes (n x n)")
    args = parser.parse_args()

    tamanhos_matriz = args.sizes
    contagens_threads = args.threads

    print("="*80)
    print("Projeto 1: DGEMM Sequencial e Paralela com Multiprocessing e NumPy")
    print("Disciplina: DEC107 - Processamento Paralelo")
    print("Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves")
    print("Data: 29/09/2025")
    print("="*80)
    print("\n| {:<10} | {:<12} | {:<15} | {:<15} | {:<15} |".format(
        "Tamanho", "Threads", "Tempo (s)", "Speedup", "Eficiência"))
    print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")

    for tamanho in tamanhos_matriz:
        A, B, _ = inicializar_matrizes(tamanho)

        # --- Sequencial ---
        inicio = time.time()
        C_seq = dgemm_sequencial(A, B)
        fim = time.time()
        tempo_seq = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
            tamanho, "1 (Seq)", tempo_seq, 1.0, 100.0))

        # --- Paralelo ---
        for n_threads in contagens_threads:
            inicio = time.time()
            C_par = dgemm_paralelo(A, B, n_threads)
            fim = time.time()
            tempo_par = fim - inicio
            speedup = tempo_seq / tempo_par
            eficiencia = (speedup / n_threads) * 100
            print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
                tamanho, n_threads, tempo_par, speedup, eficiencia))

        # --- NumPy BLAS ---
        inicio = time.time()
        C_blas = A @ B
        fim = time.time()
        tempo_blas = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15} | {:<15} |".format(
            tamanho, "BLAS", tempo_blas, "-", "-"))
        print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
