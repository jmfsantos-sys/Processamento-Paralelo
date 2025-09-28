"""
Projeto 1 — DGEMM Sequencial e Paralela com Python
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Disciplina: DEC107 — Processamento Paralelo
Data: 29/09/2025

Versões:
    - Sequencial puro (Python + Numba)
    - Paralelo multithreaded (Numba prange)
    - Paralelo híbrido (Numba prange + multiprocessing)
    - BLAS/NumPy como referência
"""

import numpy as np
import time
import numba
from numba import njit, prange, set_num_threads
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import os
import argparse

# =========================
# Funções de inicialização
# =========================
def inicializar_matrizes(n):
    """
    Inicializa A, B com valores aleatórios e C com zeros.
    """
    A = np.random.randint(-1, 2, size=(n, n), dtype=np.float64)
    B = np.random.randint(-4, 5, size=(n, n), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    return A, B, C

# =========================
# DGEMM Sequencial (Numba)
# =========================
@njit
def dgemm_sequencial(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C[i, j] = sum_
    return C

# =========================
# DGEMM Paralelo multithreaded (Numba prange)
# =========================
@njit(parallel=True)
def dgemm_paralelo(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for k in range(n):
            a_ik = A[i, k]
            for j in range(n):
                C[i, j] += a_ik * B[k, j]
    return C

# =========================
# DGEMM híbrido: Numba + multiprocessing
# =========================
def dgemm_hibrido_worker(args):
    A_slice, B = args
    n_rows = A_slice.shape[0]
    n = B.shape[0]
    C_slice = np.zeros((n_rows, n), dtype=np.float64)
    for i in range(n_rows):
        for k in range(n):
            a_ik = A_slice[i, k]
            for j in range(n):
                C_slice[i, j] += a_ik * B[k, j]
    return C_slice

def dgemm_hibrido(A, B, n_processos=None):
    if n_processos is None:
        n_processos = cpu_count()
    chunk_size = A.shape[0] // n_processos
    slices = [(A[i*chunk_size:(i+1)*chunk_size], B) for i in range(n_processos)]
    with Pool(processes=n_processos) as pool:
        C_slices = pool.map(dgemm_hibrido_worker, slices)
    return np.vstack(C_slices)

# =========================
# Testes e coleta de resultados
# =========================
def executar_testes(tamanhos, threads_list, salvar_pdf=True):
    resultados = []
    print("=================================================================================")
    print("Projeto 1: DGEMM Sequencial e Paralela com Python + Numba + Multiprocessing")
    print("Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves")
    print("Data: 29/09/2025")
    print("=================================================================================")
    print("| {:<8} | {:<8} | {:<12} | {:<10} | {:<10} | {:<10} |".format(
        "Tamanho", "Threads", "Tempo(s)", "GFLOPS", "Speedup", "Eficiência"))
    print("|----------|----------|--------------|------------|------------|------------|")

    for n in tamanhos:
        A, B, _ = inicializar_matrizes(n)
        flops = 2.0 * n**3

        # Sequencial
        t0 = time.time()
        C_seq = dgemm_sequencial(A, B)
        t1 = time.time()
        tempo_seq = t1 - t0
        gflops_seq = flops / tempo_seq / 1e9
        print("| {:<8} | {:<8} | {:<12.6f} | {:<10.3f} | {:<10} | {:<10} |".format(
            n, 1, tempo_seq, gflops_seq, "-", "-"))
        resultados.append({'n': n, 'threads':1, 'tempo':tempo_seq, 'gflops':gflops_seq, 'speedup':1.0, 'eficiencia':100.0})

        # Paralelo Numba prange
        for t in threads_list:
            set_num_threads(t)
            t0 = time.time()
            C_par = dgemm_paralelo(A, B)
            t1 = time.time()
            tempo_par = t1 - t0
            speedup = tempo_seq / tempo_par
            eficiencia = speedup / t * 100
            gflops_par = flops / tempo_par / 1e9
            print("| {:<8} | {:<8} | {:<12.6f} | {:<10.3f} | {:<10.3f} | {:<9.2f}% |".format(
                n, t, tempo_par, gflops_par, speedup, eficiencia))
            resultados.append({'n': n, 'threads':t, 'tempo':tempo_par, 'gflops':gflops_par, 'speedup':speedup, 'eficiencia':eficiencia})

        # Híbrido multiprocessing
        t0 = time.time()
        C_hyb = dgemm_hibrido(A, B)
        t1 = time.time()
        tempo_hyb = t1 - t0
        speedup_hyb = tempo_seq / tempo_hyb
        gflops_hyb = flops / tempo_hyb / 1e9
        eficiencia_hyb = speedup_hyb / cpu_count() * 100
        print("| {:<8} | {:<8} | {:<12.6f} | {:<10.3f} | {:<10.3f} | {:<9.2f}% |".format(
            n, "Hyb", tempo_hyb, gflops_hyb, speedup_hyb, eficiencia_hyb))
        resultados.append({'n': n, 'threads':'Hyb', 'tempo':tempo_hyb, 'gflops':gflops_hyb, 'speedup':speedup_hyb, 'eficiencia':eficiencia_hyb})

        # BLAS / NumPy
        t0 = time.time()
        C_blas = A @ B
        t1 = time.time()
        tempo_blas = t1 - t0
        gflops_blas = flops / tempo_blas / 1e9
        print("| {:<8} | {:<8} | {:<12.6f} | {:<10.3f} | {:<10} | {:<10} |".format(
            n, "BLAS", tempo_blas, gflops_blas, "-", "-"))
        resultados.append({'n': n, 'threads':'BLAS', 'tempo':tempo_blas, 'gflops':gflops_blas, 'speedup':'-', 'eficiencia':'-'})
        print("|----------|----------|--------------|------------|------------|------------|")

    if salvar_pdf:
        plotar_resultados(resultados, tamanhos, threads_list)
    return resultados

# =========================
# Gráficos e PDF
# =========================
def plotar_resultados(resultados, tamanhos, threads_list):
    for n in tamanhos:
        dados = [r for r in resultados if r['n']==n and isinstance(r['threads'], int)]
        ts = [r['threads'] for r in dados]
        speedups = [r['speedup'] for r in dados]
        eficiencias = [r['eficiencia'] for r in dados]

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(ts, speedups, marker='o')
        plt.title(f"Speedup - Matriz {n}x{n}")
        plt.xlabel("Threads")
        plt.ylabel("Speedup")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(ts, eficiencias, marker='o', color='orange')
        plt.title(f"Eficiência - Matriz {n}x{n}")
        plt.xlabel("Threads")
        plt.ylabel("Eficiência (%)")
        plt.ylim(0, 120)
        plt.grid(True)

        plt.tight_layout()
        filename = f"desempenho_{n}x{n}.pdf"
        plt.savefig(filename)
        print(f"Gráfico salvo em: {filename}")
        plt.close()

# =========================
# Main: argumentos dinâmicos
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGEMM Sequencial e Paralela Python")
    parser.add_argument("--max_size", type=int, default=1024, help="Tamanho máximo da matriz (até 4096)")
    parser.add_argument("--threads", type=int, nargs="+", default=[2,4,8], help="Número de threads para prange")
    args = parser.parse_args()

    tamanhos = [512, 1024, 2048, 4096]
    tamanhos = [t for t in tamanhos if t <= args.max_size]

    executar_testes(tamanhos, args.threads, salvar_pdf=True)
