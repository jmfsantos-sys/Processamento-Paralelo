# Projeto 1 - Multiplicação de Matrizes (DGEMM) Sequencial e Paralela com OpenMP

## Descrição

Este projeto implementa a multiplicação de matrizes densas (DGEMM) de forma **sequencial** e **paralela** usando **OpenMP**. O objetivo é medir e comparar o desempenho entre as versões sequencial, paralela e a referência de alta performance (BLAS).  

O projeto também gera resultados em **CSV** e um relatório em **Excel** com tabelas e gráficos de desempenho, facilitando a análise dos experimentos.

---

## Autores

- João Manoel Fidelis Santos  
- Maria Eduarda Guedes Alves  

**Disciplina:** DEC107 – Processamento Paralelo  
**Curso:** Bacharelado em Ciência da Computação  
**Data:** 27/09/2025

---

## Funcionalidades

1. **Multiplicação de matrizes:**
   - Sequencial otimizada
   - Paralela otimizada (OpenMP)
   - BLAS (referência de alta performance)
2. **Medição de:**
   - Tempo de execução médio
   - GFLOPS
   - Speedup
   - Eficiência paralela real
3. Geração de arquivos CSV com os resultados
4. **Script Python para:**
   - Ler os CSVs
   - Gerar tabelas organizadas
   - Criar gráficos de Tempo, GFLOPS, Speedup e Eficiência
   - Gerar relatório Excel completo

---

## Estrutura do Repositório



.
├── vPalV9_final.c          # Código principal com saída de log detalhada


├── vPalV6_clean_csv.c      # Código com saída CSV para análise


├── plot_dgemm.py           # Script Python para geração de relatório Excel


├── resultados.csv          # Arquivo CSV gerado pelos testes


└── README.md               # Este arquivo


---

## Passo a Passo para Execução Completa (Linux/WSL)

### 1. Atualização do Sistema e Pacotes

bash

sudo apt update

sudo apt upgrade -y

sudo apt autoremove


### 2. Instalação de Compiladores e Bibliotecas

bash

sudo apt install -y build-essential libopenblas-dev libcblas-dev libatlas-base-dev


### 3. Instalação do Python e Bibliotecas de Análise

bash

sudo apt install -y python3-pip python3-pandas python3-matplotlib python3-xlsxwriter

pip3 install openpyxl


### 4. Navegação até o Diretório do Projeto


### 5. Compilação dos Códigos com OpenMP e BLAS

bash
# Versão CSV para análise completa

gcc -O3 -fopenmp -o dgemm_csv vPalV6_clean_csv.c -lopenblas

# Versões sequencial e BLAS para comparação

gcc -O3 -fopenmp -o dgemm_seq vPalV6_clean.c

gcc -O3 -fopenmp -o dgemm_blas vPalV6_clean.c -lopenblas

gcc -O3 -fopenmp -o dgemm_seq vPalV6_clean_csv.c

gcc -O3 -fopenmp -o dgemm_blas vPalV6_clean_csv.c -lopenblas

### 6. Execução dos Programas

bash

# Gerar arquivo CSV com resultados

./dgemm_csv > resultados.csv


# Testes adicionais para log detalhado

./dgemm_seq

./dgemm_blas


### 7. Verificação dos Resultados

bash

head -n 5 resultados.csv


### 8. Geração de Relatórios com Python

bash

python3 plot_dgemm.py


> O script processa resultados.csv e gera relatorio_dgemm.xlsx com tabelas e gráficos de:
> 
>
> * Tempo de execução
> * 
> * GFLOPS
> * 
> * Speedup
> * 
> * Eficiência
> * 

---

## Exemplo de Saída

**CSV gerado:**


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
* As matrizes são inicializadas com valores aleatórios a cada execução.
* A eficiência real pode ultrapassar 100% em casos de otimizações de cache ou escalonamento de threads.
* O script Python detecta automaticamente o separador do CSV e limpa dados inválidos.

---

## Licença

Este projeto é de uso acadêmico e pode ser utilizado para estudo e experimentos em Processamento Paralelo.

