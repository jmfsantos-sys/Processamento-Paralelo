# Projeto 1 - Multiplicação de Matrizes (DGEMM) Sequencial e Paralela com OpenMP

## Descrição

Este projeto implementa a multiplicação de matrizes densas (DGEMM) de forma **sequencial** e **paralela** usando **OpenMP**. O objetivo é medir e comparar o desempenho entre as versões sequencial, paralela e a referência de alta performance (BLAS).  

O projeto também gera resultados em **CSV** e um relatório em **Excel** com tabelas e gráficos de desempenho, facilitando a análise dos experimentos.

---

## Autores

- João Manoel Fidelis Santos  
- Maria Eduarda Guedes Alves  

Disciplina: DEC107 – Processamento Paralelo  
Curso: Bacharelado em Ciência da Computação  

Data: 27/09/2025

---

## Funcionalidades

1. Multiplicação de matrizes:
   - Sequencial otimizada
   - Paralela otimizada (OpenMP)
   - BLAS (referência de alta performance)
2. Medição de:
   - Tempo de execução médio
   - GFLOPS
   - Speedup
   - Eficiência paralela real
3. Geração de arquivos CSV com os resultados
4. Script Python para:
   - Ler os CSVs
   - Gerar tabelas organizadas
   - Criar gráficos de Tempo, GFLOPS, Speedup e Eficiência
   - Gerar relatório Excel completo

---

## Estrutura do Repositório



.
├── vPalV9_final.c          # Código principal com saída de log detalhada
├── vPalV6_clean_csv.c      # Código com saída CSV
├── resultados.csv          # Arquivo CSV gerado pelos testes
├── analisar_resultados.py  # Script Python para geração de relatório Excel
└── README.md               # Este arquivo



---

## Compilação e Execução

### Compilação com GCC (Linux / WSL)

bash
gcc -fopenmp -O3 vPalV6_clean_csv.c -o dgemm -lblas


### Execução

bash
./dgemm > resultados.csv


> ⚠️ Certifique-se de que a biblioteca BLAS (cblas) está instalada no seu sistema.

---

## Geração de Relatório

O script Python analisar_resultados.py processa o CSV gerado e produz:

* Tabela organizada de desempenho por tamanho de matriz e número de threads
* Gráficos de:

  * Tempo de execução
  * GFLOPS
  * Speedup
  * Eficiência
* Relatório Excel completo (relatorio_dgemm.xlsx) com tabelas e gráficos incorporados

### Execução do script Python

bash
python plot_dgemm.py


---

## Exemplo de Saída

**CSV:**


N,Threads,Tempo,GFLOPS,Speedup,Eficiencia
512,1,0.123,4.56,1.00,100.00
512,2,0.067,8.38,1.83,91.50
...


**Gráficos gerados no Excel:**

* Comparação de tempo por número de threads
* Desempenho em GFLOPS
* Speedup obtido com paralelismo
* Eficiência percentual das execuções paralelas

---

## Observações

* O código paraleliza o laço externo da multiplicação (i) usando OpenMP.
* As matrizes são inicializadas com valores aleatórios para cada execução.
* A eficiência real pode ultrapassar 100% em casos de otimizações de cache ou escalonamento de threads.
* O script Python detecta automaticamente o separador do CSV e limpa dados inválidos.

---

## Licença

Este projeto é de uso acadêmico e pode ser utilizado para estudo e experimentos em Processamento Paralelo.
