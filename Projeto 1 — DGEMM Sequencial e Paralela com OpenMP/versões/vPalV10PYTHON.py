"""
================================================================================
Projeto — DGEMM Sequencial e Paralela Híbrida Python
Disciplina: DEC107 — Processamento Paralelo
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Data: 27/09/2025
Objetivo:
- DGEMM híbrido: Numba multithread + Multiprocessing + Cache Blocking
- Geração automática de tabela comparativa (CSV)
- Gráficos de Speedup e Eficiência prontos para PDF
================================================================================
"""

import numpy as np
import time
from numba import njit, prange
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
# DGEMM Sequencial
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
                            sum_ += A[i, k] * B[k, j]
                        C[i, j] += sum_

# -------------------------------
# Multiprocessing puro
# -------------------------------
def worker_multiproc(start, end, A, B, shared_C):
    n = A.shape[0]
    C_np = np.frombuffer(shared_C.get_obj()).reshape(n, n)
    for i in range(start, end):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C_np[i, j] = sum_

def dgemm_multiprocessing(A, B, C, n_threads):
    n = A.shape[0]
    shared_C = mp.Array('d', n*n)
    chunk_size = n // n_threads
    processes = []
    for t in range(n_threads):
        start = t*chunk_size
        end = (t+1)*chunk_size if t != n_threads-1 else n
        p = mp.Process(target=worker_multiproc, args=(start, end, A, B, shared_C))
        processes.append(p)
        p.start()
    for p in processes: p.join()
    C[:] = np.frombuffer(shared_C.get_obj()).reshape(n, n)

# -------------------------------
# Híbrido: Numba + Multiprocessing + cache blocking
# -------------------------------
def worker_hybrid(start, end, A, B, shared_C, block_size):
    n = A.shape[0]
    C_np = np.frombuffer(shared_C.get_obj()).reshape(n, n)
    dgemm_numba_blocked(A[start:end, :], B, C_np[start:end, :], block_size)

def dgemm_hybrid(A, B, C, n_threads, block_size=64):
    n = A.shape[0]
    shared_C = mp.Array('d', n*n)
    chunk_size = n // n_threads
    processes = []
    for t in range(n_threads):
        start = t*chunk_size
        end = (t+1)*chunk_size if t != n_threads-1 else n
        p = mp.Process(target=worker_hybrid, args=(start, end, A, B, shared_C, block_size))
        processes.append(p)
        p.start()
    for p in processes: p.join()
    C[:] = np.frombuffer(shared_C.get_obj()).reshape(n, n)

# -------------------------------
# Experimentos automáticos e tabela comparativa
# -------------------------------
def run_experiments(max_size=1024, thread_counts=[2,4,8], block_size=64):
    sizes = [512, 1024, 2048, 4096]
    sizes = [s for s in sizes if s <= max_size]

    print("| {:>8} | {:>8} | {:>12} | {:>10} | {:>10} | {:>10} |".format(
        "Tamanho","Threads","Tempo(s)","GFLOPS","Speedup","Eficiência"))
    print("|----------|----------|--------------|------------|------------|------------|")

    results = []

    for n in sizes:
        A = alocar_matriz(n); B = alocar_matriz(n); C = alocar_matriz(n)
        inicializa_matrizes(A,B,C)
        flops = 2.0*n**3

        # Sequencial
        start = time.time()
        dgemm_seq(A,B,C)
        dt_seq = time.time() - start
        gflops_seq = flops/dt_seq/1e9
        results.append((n,1,dt_seq,gflops_seq,"-","-"))
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10} | {:>10} |".format(
            n,1,dt_seq,gflops_seq,"-","-"))

        # Numba
        C[:] = 0
        start=time.time()
        dgemm_numba_blocked(A,B,C,block_size)
        dt_numba = time.time()-start
        gflops_numba = flops/dt_numba/1e9
        speedup = dt_seq/dt_numba
        eficiencia = speedup/mp.cpu_count()*100
        results.append((n, mp.cpu_count(), dt_numba, gflops_numba, speedup, eficiencia))
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
            n, mp.cpu_count(), dt_numba, gflops_numba, speedup, eficiencia))

        # Multiprocessing puro
        for n_threads in thread_counts:
            C[:] = 0
            start=time.time()
            dgemm_multiprocessing(A,B,C,n_threads)
            dt_mp = time.time()-start
            gflops_mp = flops/dt_mp/1e9
            speedup = dt_seq/dt_mp
            eficiencia = speedup/n_threads*100
            results.append((n,n_threads,dt_mp,gflops_mp,speedup,eficiencia))
            print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
                n,n_threads,dt_mp,gflops_mp,speedup,eficiencia))

        # Híbrido
        for n_threads in thread_counts:
            C[:] = 0
            start=time.time()
            dgemm_hybrid(A,B,C,n_threads,block_size)
            dt_h = time.time()-start
            gflops_h = flops/dt_h/1e9
            speedup = dt_seq/dt_h
            eficiencia = speedup/n_threads*100
            results.append((n,n_threads,dt_h,gflops_h,speedup,eficiencia))
            print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
                n,n_threads,dt_h,gflops_h,speedup,eficiencia))

    return results

# -------------------------------
# Tabela comparativa CSV
# -------------------------------
def gerar_tabela(results, filename="dgmm_comparativa.csv"):
    df = pd.DataFrame(results, columns=["Tamanho","Threads","Tempo(s)","GFLOPS","Speedup","Eficiência(%)"])
    df.to_csv(filename,index=False)
    print(f"Tabela comparativa salva em '{filename}'")

# -------------------------------
# Gráfico PDF
# -------------------------------
def plot_comparativo(results, filename="dgmm_comparativo.pdf"):
    df = pd.DataFrame(results, columns=["Tamanho","Threads","Tempo(s)","GFLOPS","Speedup","Eficiência(%)"])
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df,x="Tamanho",y="Tempo(s)",hue="Threads",marker="o")
    plt.title("DGEMM Python: Comparativo de todas as versões")
    plt.xlabel("Tamanho da matriz")
    plt.ylabel("Tempo (s)")
    plt.legend(title="Threads")
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico comparativo salvo em '{filename}'")

# -------------------------------
# Main
# -------------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--threads", type=int, nargs='+', default=[2,4,8])
    parser.add_argument("--block_size", type=int, default=64)
    args = parser.parse_args()

    results = run_experiments(max_size=args.max_size, thread_counts=args.threads, block_size=args.block_size)
    gerar_tabela(results)
    plot_comparativo(results)
    print("\nExperimentos concluídos. Gráficos e tabela comparativa gerados.")
