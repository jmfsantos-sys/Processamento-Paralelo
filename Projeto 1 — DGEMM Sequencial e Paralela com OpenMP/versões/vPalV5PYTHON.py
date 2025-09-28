"""
Projeto 1 — DGEMM Sequencial e Paralela com Python + Numba + Matplotlib
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Disciplina: DEC107 — Processamento Paralelo
Data: 29/09/2025

Descrição:
    - Testa multiplicação de matrizes quadradas (DGEMM) sequencial, paralela e BLAS.
    - Varia tamanhos de matriz e número de threads paralelas.
    - Imprime tabela de resultados e gera gráficos de Speedup e Eficiência.
"""

import numpy as np
import time
import numba
from numba import njit, prange
import matplotlib.pyplot as plt

# -----------------------------------------
# Inicialização de matrizes
# -----------------------------------------
def inicializar_matrizes(dimensao):
    """
    Inicializa matrizes A, B com valores aleatórios e C com zeros.
    """
    A = np.random.randint(-1, 2, size=(dimensao, dimensao), dtype=np.float64)
    B = np.random.randint(-4, 5, size=(dimensao, dimensao), dtype=np.float64)
    C = np.zeros((dimensao, dimensao), dtype=np.float64)
    return A, B, C

# -----------------------------------------
# DGEMM Sequencial
# -----------------------------------------
@njit
def dgemm_sequencial_numba(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma += A[i, k] * B[k, j]
            C[i, j] = soma
    return C

# -----------------------------------------
# DGEMM Paralelo
# -----------------------------------------
@njit(parallel=True)
def dgemm_paralelo_numba(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for k in range(n):
            a_ik = A[i, k]
            for j in range(n):
                C[i, j] += a_ik * B[k, j]
    return C

# -----------------------------------------
# Execução de testes
# -----------------------------------------
def executar_testes(tamanhos_matriz, threads_list):
    """
    Executa DGEMM sequencial, paralela (vários threads) e BLAS para
    vários tamanhos de matriz e coleta resultados.
    """
    resultados = []

    print("=================================================================================")
    print("  Projeto 1: DGEMM Sequencial e Paralela com Python + Numba")
    print("  Disciplina: DEC107 - Processamento Paralelo")
    print("  Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves")
    print("  Data: 29/09/2025")
    print("=================================================================================")
    print("\n| {:<10} | {:<12} | {:<15} | {:<15} | {:<15} |".format("Tamanho", "Threads", "Tempo (s)", "Speedup", "Eficiência"))
    print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")

    for tamanho in tamanhos_matriz:
        A, B, _ = inicializar_matrizes(tamanho)

        # --- Sequencial ---
        inicio = time.time()
        C_seq = dgemm_sequencial_numba(A, B)
        fim = time.time()
        tempo_seq = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f} |".format(
            tamanho, "1 (Seq)", tempo_seq, 1.0, 100.0
        ))
        resultados.append({'tamanho': tamanho, 'threads': 1, 'tempo': tempo_seq, 'speedup': 1.0, 'eficiencia': 100.0})

        # --- Paralelo ---
        for n_threads in threads_list:
            numba.set_num_threads(n_threads)
            _, _, C_par = inicializar_matrizes(tamanho)
            inicio = time.time()
            C_par = dgemm_paralelo_numba(A, B)
            fim = time.time()
            tempo_par = fim - inicio
            speedup = tempo_seq / tempo_par
            eficiencia = (speedup / n_threads) * 100
            print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f} |".format(
                tamanho, n_threads, tempo_par, speedup, eficiencia
            ))
            resultados.append({'tamanho': tamanho, 'threads': n_threads, 'tempo': tempo_par, 'speedup': speedup, 'eficiencia': eficiencia})

        # --- BLAS / NumPy ---
        inicio = time.time()
        C_blas = A @ B
        fim = time.time()
        tempo_blas = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15} | {:<15} |".format(
            tamanho, "BLAS", tempo_blas, "-", "-"
        ))
        print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
        resultados.append({'tamanho': tamanho, 'threads': 'BLAS', 'tempo': tempo_blas, 'speedup': '-', 'eficiencia': '-'})

    return resultados

# -----------------------------------------
# Gráficos
# -----------------------------------------
def plotar_resultados(resultados, tamanhos_matriz, threads_list):
    """
    Gera gráficos de Speedup e Eficiência para cada tamanho de matriz.
    """
    for tamanho in tamanhos_matriz:
        dados = [r for r in resultados if r['tamanho'] == tamanho and r['threads'] != 'BLAS']
        threads = [r['threads'] for r in dados]
        speedups = [r['speedup'] for r in dados]
        eficiencias = [r['eficiencia'] for r in dados]

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(threads, speedups, marker='o')
        plt.title(f"Speedup - Matriz {tamanho}x{tamanho}")
        plt.xlabel("Número de Threads")
        plt.ylabel("Speedup")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(threads, eficiencias, marker='o', color='orange')
        plt.title(f"Eficiência - Matriz {tamanho}x{tamanho}")
        plt.xlabel("Número de Threads")
        plt.ylabel("Eficiência (%)")
        plt.ylim(0, 120)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# -----------------------------------------
# Execução principal
# -----------------------------------------
if __name__ == "__main__":
    # Tamanhos grandes como no seu código original
    tamanhos_matriz = [512, 1024, 2048]  # 4096 pode demorar muito dependendo do PC
    threads_list = [2, 4, 8]

    resultados = executar_testes(tamanhos_matriz, threads_list)
    plotar_resultados(resultados, tamanhos_matriz, threads_list)
