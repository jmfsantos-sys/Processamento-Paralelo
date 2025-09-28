import numpy as np
import time
from numba import njit, prange
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# Funções de multiplicação
# ---------------------------
def alocar_matriz(n):
    return np.zeros((n, n), dtype=np.float64)

def inicializa_matrizes(A, B, C):
    A[:] = np.random.randint(-1, 2, size=A.shape)
    B[:] = np.random.randint(-4, 5, size=B.shape)
    C[:] = 0.0

def dgemm_seq(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C[i, j] = sum_

@njit(parallel=True, fastmath=True)
def dgemm_numba(A, B, C):
    n = A.shape[0]
    for i in prange(n):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C[i, j] = sum_

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
    chunk_size = n // n_threads
    shared_C = mp.Array('d', n*n)
    processes = []
    for t in range(n_threads):
        start = t * chunk_size
        end = (t+1) * chunk_size if t != n_threads-1 else n
        p = mp.Process(target=worker_multiproc, args=(start, end, A, B, shared_C))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    C[:] = np.frombuffer(shared_C.get_obj()).reshape(n, n)

def dgemm_hibrido(A, B, C, n_threads):
    """ Híbrido: Numba multithread + Multiprocessing """
    n = A.shape[0]
    chunk_size = n // n_threads
    shared_C = mp.Array('d', n*n)
    processes = []
    for t in range(n_threads):
        start = t * chunk_size
        end = (t+1) * chunk_size if t != n_threads-1 else n
        p = mp.Process(target=dgemm_numba_submatriz, args=(start, end, A, B, shared_C))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    C[:] = np.frombuffer(shared_C.get_obj()).reshape(n, n)

@njit(parallel=True, fastmath=True)
def dgemm_numba_submatriz(start, end, A, B, shared_C):
    n = A.shape[0]
    C_np = np.frombuffer(shared_C.get_obj()).reshape(n, n)
    for i in prange(start, end):
        for j in range(n):
            sum_ = 0.0
            for k in range(n):
                sum_ += A[i, k] * B[k, j]
            C_np[i, j] = sum_

# ---------------------------
# Experimentos automáticos
# ---------------------------
def run_experiments(max_size=4096, thread_counts=[2,4,8]):
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
        flops = 2*n**3
        gflops_seq = flops/dt_seq/1e9
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10} | {:>10} |".format(
            n, 1, dt_seq, gflops_seq, "-", "-"))
        results.append((n,1,dt_seq,gflops_seq))

        # Numba multithread
        C[:] = 0
        start = time.time()
        dgemm_numba(A, B, C)
        dt_numba = time.time()-start
        gflops_numba = flops/dt_numba/1e9
        speedup = dt_seq/dt_numba
        eficiencia = speedup/mp.cpu_count()*100
        print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
            n, mp.cpu_count(), dt_numba, gflops_numba, speedup, eficiencia))
        results.append((n,mp.cpu_count(),dt_numba,gflops_numba,speedup,eficiencia))

        # Multiprocessing puro
        for n_threads in thread_counts:
            C[:] = 0
            start = time.time()
            dgemm_multiprocessing(A, B, C, n_threads)
            dt_mp = time.time()-start
            gflops_mp = flops/dt_mp/1e9
            speedup = dt_seq/dt_mp
            eficiencia = speedup/n_threads*100
            print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
                n, n_threads, dt_mp, gflops_mp, speedup, eficiencia))
            results.append((n,n_threads,dt_mp,gflops_mp,speedup,eficiencia))

        # Híbrido Numba + Multiprocessing
        for n_threads in thread_counts:
            C[:] = 0
            start = time.time()
            dgemm_hibrido(A,B,C,n_threads)
            dt_hibrido = time.time()-start
            gflops_h = flops/dt_hibrido/1e9
            speedup = dt_seq/dt_hibrido
            eficiencia = speedup/n_threads*100
            print("| {:>8} | {:>8} | {:>12.6f} | {:>10.3f} | {:>10.2f} | {:>9.2f}% |".format(
                n,n_threads,dt_hibrido,gflops_h,speedup,eficiencia))
            results.append((n,n_threads,dt_hibrido,gflops_h,speedup,eficiencia))

    return results

# ---------------------------
# Geração de gráfico PDF
# ---------------------------
def plot_results(results, filename="dgmm_report.pdf"):
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
    plt.title("DGEMM Python: Tempo de execução")
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ---------------------------
# Main
# ---------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=1024, help="Tamanho máximo da matriz")
    parser.add_argument("--threads", type=int, nargs='+', default=[2,4,8], help="Threads para multiprocessing")
    args = parser.parse_args()
    results = run_experiments(max_size=args.max_size, thread_counts=args.threads)
    plot_results(results)
    print("\nExperimentos concluídos. Gráfico salvo em 'dgmm_report.pdf'.")
