import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# --- Função para detectar separador automaticamente ---
def detect_separator(file_path, possible_seps=[',', ';', '\t']):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            for sep in possible_seps:
                if sep in line:
                    return sep
    return ','  # default

# --- Carrega CSV robustamente ---
csv_file = "resultados.csv"
sep = detect_separator(csv_file)
df = pd.read_csv(
    csv_file,
    sep=sep,
    comment="#",  # ignora comentários
    header=None,
    names=["N", "Threads", "Tempo", "GFLOPS", "Speedup", "Eficiencia"]
)

# --- Converte colunas numéricas ---
for col in ["N", "Threads", "Tempo", "GFLOPS", "Speedup", "Eficiencia"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Remove linhas inválidas ---
df = df.dropna(subset=["N", "Threads", "Tempo", "GFLOPS"])
df["N"] = df["N"].astype(int)
df["Threads"] = df["Threads"].astype(int)

# --- Calcula Speedup e Eficiência se necessário ---
if df["Speedup"].isna().all():
    df["Speedup"] = df.groupby("N")["Tempo"].transform(lambda x: x.iloc[0] / x)
if df["Eficiencia"].isna().all():
    df["Eficiencia"] = df["Speedup"] / df["Threads"] * 100

# --- Lista de tamanhos e threads ---
tamanhos = sorted(df["N"].unique())
threads = sorted(df["Threads"].unique())

# --- Criar tabela bonita ---
tabelas = []
for n in tamanhos:
    subset = df[df["N"] == n].set_index("Threads")
    temp = pd.DataFrame({
        "Métrica": ["Tempo (s)", "GFLOPS", "Speedup", "Eficiência (%)"],
        **{thr: [subset.loc[thr, "Tempo"],
                 subset.loc[thr, "GFLOPS"],
                 subset.loc[thr, "Speedup"],
                 subset.loc[thr, "Eficiencia"]] for thr in threads}
    })
    temp.insert(0, "N", n)
    tabelas.append(temp)

tabela_final = pd.concat(tabelas, ignore_index=True)
tabela_final = tabela_final[["N", "Métrica"] + threads]

# --- Função para criar gráfico em memória ---
def criar_grafico(y_col, ylabel, title):
    plt.figure(figsize=(6,4))
    for n in tamanhos:
        subset = df[df["N"] == n]
        plt.plot(subset["Threads"], subset[y_col], marker="o", label=f"N={n}")
    plt.xlabel("Número de Threads")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    plt.close()
    img_data.seek(0)
    return img_data

graficos = {
    "Tempo de Execução": criar_grafico("Tempo", "Tempo (s)", "Tempo de Execução"),
    "GFLOPS": criar_grafico("GFLOPS", "GFLOPS", "Desempenho (GFLOPS)"),
    "Speedup": criar_grafico("Speedup", "Speedup", "Speedup Paralelo"),
    "Eficiência": criar_grafico("Eficiencia", "Eficiência (%)", "Eficiência do Paralelismo")
}

# --- Gerar relatório Excel completo ---
with pd.ExcelWriter("relatorio_dgemm.xlsx", engine="xlsxwriter") as writer:
    # Escreve tabela bonita
    tabela_final.to_excel(writer, sheet_name="Tabela Bonita", index=False)
    
    # Acessa worksheet
    worksheet = writer.sheets["Tabela Bonita"]
    
    # Inserir gráficos abaixo da tabela
    start_row = len(tabela_final) + 3
    for i, (title, img_data) in enumerate(graficos.items()):
        worksheet.insert_image(start_row + i*20, 0, title + ".png", {'image_data': img_data})

print("✅ Relatório Excel completo gerado: relatorio_dgemm.xlsx")
