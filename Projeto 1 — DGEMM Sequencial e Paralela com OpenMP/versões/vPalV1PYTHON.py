"""
Projeto 1 — DGEMM Sequencial e Paralela
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Disciplina: DEC107 — Processamento Paralelo
Data: 29/09/2025
"""

import numpy as np
import time
from numba import njit, prange

# ----------------------
# Funções principais
# ----------------------

def inicializar_matrizes(dimensao):
    """Cria e inicializa matrizes A, B e C."""
    A = np.random.randint(-1, 2, size=(dimensao, dimensao), dtype=np.float64)
    B = np.random.randint(-4, 5, size=(dimensao, dimensao), dtype=np.float64)
    C = np.zeros((dimensao, dimensao), dtype=np.float64)
    return A, B, C

# --- Multiplicação sequencial ---
@njit
def dgemm_sequencial(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma += A[i, k] * B[k, j]
            C[i, j] = soma

# --- Multiplicação paralela com Numba ---
@njit(parallel=True)
def dgemm_paralelo(A, B, C):
    n = A.shape[0]
    for i in prange(n):  # loop paralelo
        for k in range(n):
            a_ik = A[i, k]
            for j in range(n):
                C[i, j] += a_ik * B[k, j]

# ----------------------
# Script principal
# ----------------------

if __name__ == "__main__":
    tamanhos_matriz = [512, 1024, 2048, 4096]
    contagens_threads = [2, 4, 8]  # Controla via Numba: NUMBA_NUM_THREADS

    print("="*80)
    print("Projeto 1: DGEMM Sequencial e Paralela com Numba e NumPy")
    print("Disciplina: DEC107 - Processamento Paralelo")
    print("Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves")
    print("Data: 29/09/2025")
    print("="*80)
    print("\n| {:<10} | {:<12} | {:<15} | {:<15} | {:<15} |".format(
        "Tamanho", "Threads", "Tempo (s)", "Speedup", "Eficiência"))
    print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")

    for tamanho in tamanhos_matriz:
        # Inicializa matrizes
        A, B, C = inicializar_matrizes(tamanho)

        # --- Sequencial ---
        inicio = time.time()
        dgemm_sequencial(A, B, C)
        fim = time.time()
        tempo_seq = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
            tamanho, "1 (Seq)", tempo_seq, 1.0, 100.0))

        # --- Paralelo ---
        for n_threads in contagens_threads:
            import os
            os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
            C.fill(0.0)  # reseta matriz C

            inicio = time.time()
            dgemm_paralelo(A, B, C)
            fim = time.time()
            tempo_par = fim - inicio

            speedup = tempo_seq / tempo_par
            eficiencia = (speedup / n_threads) * 100
            print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
                tamanho, n_threads, tempo_par, speedup, eficiencia))

        # --- Usando NumPy/BLAS ---
        C.fill(0.0)
        inicio = time.time()
        C = A @ B
        fim = time.time()
        tempo_blas = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15} | {:<15} |".format(
            tamanho, "BLAS", tempo_blas, "-", "-"))
        print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
