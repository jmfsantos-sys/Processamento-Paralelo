"""
Projeto 1 — DGEMM Sequencial e Paralela com Numba + Multithreading
Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves
Disciplina: DEC107 — Processamento Paralelo
Data: 29/09/2025
Descrição:
    Implementa a multiplicação de matrizes (DGEMM) em Python utilizando Numba.
    - dgemm_sequencial_numba: versão sequencial pura.
    - dgemm_paralelo_numba: versão paralela usando Numba e prange (multithreading).
    - Também compara com NumPy/BLAS como referência.
"""

# -------------------------------
# Importações de bibliotecas
# -------------------------------
import numpy as np        # Para manipulação de matrizes
import time               # Para medir tempo de execução
from numba import njit, prange  # Para acelerar funções e permitir paralelismo

# -------------------------------
# Inicialização de matrizes
# -------------------------------
def inicializar_matrizes(dimensao):
    """
    Cria e inicializa três matrizes quadradas A, B e C:
      - A e B: valores aleatórios inteiros
      - C: matriz de zeros
    Parâmetros:
        dimensao (int): tamanho da matriz (n x n)
    Retorna:
        A, B, C (np.ndarray): matrizes inicializadas
    """
    # np.random.randint gera valores inteiros em [low, high)
    A = np.random.randint(-1, 2, size=(dimensao, dimensao), dtype=np.float64)
    B = np.random.randint(-4, 5, size=(dimensao, dimensao), dtype=np.float64)
    C = np.zeros((dimensao, dimensao), dtype=np.float64)
    return A, B, C

# -------------------------------
# Multiplicação sequencial com Numba
# -------------------------------
@njit
def dgemm_sequencial_numba(A, B):
    """
    Multiplicação de matrizes quadradas A e B de forma sequencial.
    njit: JIT compilation para acelerar a função.
    Parâmetros:
        A, B: matrizes numpy 2D
    Retorna:
        C: matriz resultante A * B
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)  # inicializa a matriz resultado
    # Laço triplo clássico (i-j-k)
    for i in range(n):
        for j in range(n):
            soma = 0.0
            for k in range(n):
                soma += A[i, k] * B[k, j]  # multiplicação e acumulação
            C[i, j] = soma  # armazena o resultado na posição (i,j)
    return C

# -------------------------------
# Multiplicação paralela com Numba e prange
# -------------------------------
@njit(parallel=True)  # habilita paralelismo automático
def dgemm_paralelo_numba(A, B):
    """
    Multiplicação de matrizes quadradas A e B utilizando multithreading.
    - njit(parallel=True) permite que o Numba paralelize laços.
    - prange é usado para laço paralelizável.
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)  # matriz resultado
    # Laço externo em prange: Numba divide iterações entre threads
    for i in prange(n):
        for k in range(n):
            a_ik = A[i, k]  # lê valor de A[i,k] uma vez
            for j in range(n):
                # ordem ikj melhora localidade de memória/cache
                C[i, j] += a_ik * B[k, j]
    return C

# -------------------------------
# Função principal (experimentos)
# -------------------------------
if __name__ == "__main__":
    # Tamanhos das matrizes para teste
    tamanhos_matriz = [512, 1024, 2048]  # você pode aumentar ou diminuir
    # Número de threads será definido automaticamente pelo Numba
    print("="*80)
    print("Projeto 1: DGEMM Sequencial e Paralela com Numba + prange")
    print("Disciplina: DEC107 - Processamento Paralelo")
    print("Autores: João Manoel Fidelis Santos e Maria Eduarda Guedes Alves")
    print("Data: 29/09/2025")
    print("="*80)

    # Cabeçalho da tabela de resultados
    print("\n| {:<10} | {:<12} | {:<15} | {:<15} | {:<15} |".format(
        "Tamanho", "Versão", "Tempo (s)", "Speedup", "Eficiência"))
    print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")

    # Loop sobre tamanhos de matriz
    for tamanho in tamanhos_matriz:
        # Inicializa matrizes A, B e C
        A, B, _ = inicializar_matrizes(tamanho)

        # -------------------
        # DGEMM Sequencial
        # -------------------
        inicio = time.time()
        C_seq = dgemm_sequencial_numba(A, B)  # chamada da função sequencial
        fim = time.time()
        tempo_seq = fim - inicio
        # Exibe resultados da versão sequencial
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
            tamanho, "Sequencial", tempo_seq, 1.0, 100.0))

        # -------------------
        # DGEMM Paralelo
        # -------------------
        inicio = time.time()
        C_par = dgemm_paralelo_numba(A, B)  # chamada da função paralela
        fim = time.time()
        tempo_par = fim - inicio
        speedup = tempo_seq / tempo_par
        # eficiência = speedup / threads * 100%
        # Numba usa número de threads automático (num_threads)
        num_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        eficiencia = (speedup / num_threads) * 100
        # Exibe resultados da versão paralela
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15.2f} | {:<15.2f}% |".format(
            tamanho, f"Paralelo ({num_threads})", tempo_par, speedup, eficiencia))

        # -------------------
        # NumPy / BLAS (referência)
        # -------------------
        inicio = time.time()
        C_blas = A @ B  # operador @ chama BLAS otimizado
        fim = time.time()
        tempo_blas = fim - inicio
        print("| {:<10} | {:<12} | {:<15.6f} | {:<15} | {:<15} |".format(
            tamanho, "BLAS", tempo_blas, "-", "-"))

        # Linha de separação
        print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
