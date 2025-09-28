"""
================================================================================
Projeto 1 — DGEMM Sequencial e Paralela Híbrida Python
Disciplina: DEC107 — Processamento Paralelo
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Data: 27/09/2025
Objetivo:
- Implementar DGEMM híbrido (Numba + Multiprocessing + Cache Blocking)
- Gerar relatório tabulado estilo printf
- Gerar gráficos de speedup e eficiência prontos para PDF
================================================================================
"""

import numpy as np
import time
from numba import njit, prange
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse

# -------------------------------
# Funções utilitárias
# -------------------------------

def alocar_matriz(n):
    return np.zeros((n, n), dtype=np.float64)

def inicializa_matrizes(A, B, C):
    A[:] = np.random.randint(-1, 2, size=A.shape)
    B[:] = np.random.randint(-4, 5, size=B.shape)
    C[:] = 0.0

# -------------------------------
# DGEMM sequencial
# -------------------------------

def dgemm_seq(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C[i, j] = sum_

# -------------------------------
# DGEMM Numba multithread + cache blocking
# -------------------------------

@njit(parallel=True, fastmath=True)
def dgemm_numba_blocked(A, B, C, block_size=64):
    n = A.shape[0]
    for ii in prange(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                i_max = min(ii+block_size, n)
                j_max = min(jj+block_size, n)
                k_max = min(kk+block_size, n)
                for i in range(ii, i_max):
                    for j in range(jj, j_max):
                        sum_ = 0.0
                        for k in range(kk, k_max):
                            sum_ += A[i, k]*B[k, j]
                        C[i, j] += sum_

# -------------------------------
# Funções para multiprocessing híbrido
# -------------------------------

def worker_hybrid(start, end, A, B, shared_C, block_size):
    n = A.shape[0]
    C_np = np.frombuffer(shared_C.get_obj()).reshape(n, n)
    dgemm_numba_blocked(A[start:end, :], B, C_np[start:end, :], block_size)

def dgemm_hybrid(A, B, C, n_threads, block_size=64):
    n = A.shape[0]
    chunk_size = n // n_threads
    shared_C = mp.Array('d', n*n)
    processes = []
    for t in range(n_threads):
        start = t * chunk_size
        end = (t+1) * chunk_size if t != n_threads-1 else n
        p = mp.Process(target=worker_hybrid, args=(start, end, A, B, shared_C, block_size))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    C[:] = np.frombuffer(shared_C.get_obj()).reshape(n, n)

# -------------------------------
# Experimentos automáticos
# -------------------------------

def run_experiments(max_size=4096, thread_counts=[2,4,8], block_size=64):
    sizes = [512, 1024, 2048, 4096]
    sizes = [s for s in sizes if s <= max_size]

    print("| {:>8} | {:>8} | {:>12} | {:>10} | {:>10} | {:>10} |".format(
        "Tamanho", "Threads", "Tempo(s)", "GFLOPS", "Speedup", "Eficiência"))
    print("|----------|----------|--------------|------------|------------|------------|")

    results = []

    for n in sizes:
        A = alocar_matriz(n)
        B = alocar_matriz(n)
        C = alocar_matriz(n)
        inicializa_matrizes(A, B, C)

        # Sequencial
        start = time.time()
        dgemm_seq(A, B, C)
        dt_seq = time.time() - start
        flops = 2.0 * n**3
        gflops_seq = flops/dt_seq/1e9
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10} | {:>10} |".format(
            n, 1, dt_seq, gflops_seq, "-", "-"))
        results.append((n, 1, dt_seq, gflops_seq))

        # Numba multithread + cache blocking
        C[:] = 0
        start = time.time()
        dgemm_numba_blocked(A, B, C, block_size)
        dt_numba = time.time()-start
        gflops_numba = flops/dt_numba/1e9
        speedup = dt_seq/dt_numba
        eficiencia = speedup / mp.cpu_count() * 100
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
            n, mp.cpu_count(), dt_numba, gflops_numba, speedup, eficiencia))
        results.append((n, mp.cpu_count(), dt_numba, gflops_numba, speedup, eficiencia))

        # Híbrido Numba + Multiprocessing
        for n_threads in thread_counts:
            C[:] = 0
            start = time.time()
            dgemm_hybrid(A, B, C, n_threads, block_size)
            dt_hybrid = time.time()-start
            gflops_h = flops/dt_hybrid/1e9
            speedup = dt_seq/dt_hybrid
            eficiencia = speedup/n_threads*100
            print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
                n, n_threads, dt_hybrid, gflops_h, speedup, eficiencia))
            results.append((n,n_threads,dt_hybrid,gflops_h,speedup,eficiencia))

    return results

# -------------------------------
# Função para gráficos PDF
# -------------------------------

def plot_results(results, filename="dgmm_hybrid_report.pdf"):
    import seaborn as sns
    sns.set(style="whitegrid")
    sizes = sorted(set([r[0] for r in results]))
    thread_counts = sorted(set([r[1] for r in results]))
    plt.figure(figsize=(10,6))
    for n_threads in thread_counts:
        times = [r[2] for r in results if r[1]==n_threads]
        plt.plot(sizes, times, marker='o', label=f"{n_threads} threads")
    plt.xlabel("Tamanho da matriz")
    plt.ylabel("Tempo (s)")
    plt.title("DGEMM Python Híbrido: Tempo de execução")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# -------------------------------
# Main
# -------------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=1024, help="Tamanho máximo da matriz")
    parser.add_argument("--threads", type=int, nargs='+', default=[2,4,8], help="Threads para multiprocessing")
    parser.add_argument("--block_size", type=int, default=64, help="Tamanho do bloco para cache blocking")
    args = parser.parse_args()

    results = run_experiments(max_size=args.max_size, thread_counts=args.threads, block_size=args.block_size)
    plot_results(results, filename="dgmm_hybrid_report.pdf")
    print("\nExperimentos concluídos. Gráfico salvo em 'dgmm_hybrid_report.pdf'.")
