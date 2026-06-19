import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- 1. LEITURA E PRÉ-PROCESSAMENTO ---
df = pd.read_csv(r'Bootcamp-DataScience-2026\fundo_amanha_dados_mock_2026.csv', sep=',', encoding='latin-1')

df = df.drop_duplicates().dropna()
df = df[df['temperaturamediareal'] >= 1000]
df = df[df['sugestaomodelolegado'] >= -100]

# Engenharias de features base
df['c_med'] = (df['c_min'] + df['c_max']) / 2
df['c_diff'] = df['c_max'] - df['c_min']
df['progresso_sequencia'] = df['sequencia'] / df['sequenciatotal']

df = pd.get_dummies(df, columns=['qualidade'], prefix='qualidade')

# Removendo colunas desnecessárias ou que causam vazamento de dados
df = df.drop(['acoatual', 'c_min', 'c_max', 'corrida', 'sugestaomodelolegado', 'temperaturamediareal'], axis=1)

# Isolando a variável alvo
y = df['temperaturasaidafp']
df = df.drop(['temperaturasaidafp'], axis=1)

# --- 2. DEFINIÇÃO DOS BLOCOS (VARIÁVEIS) ---
colunas = df.columns.tolist()

quimica_cols = [c for c in colunas if 'qualidade' in c] + ['al_min', 'n_min', 's_min', 'c_med', 'c_diff', 'temperaturaliquidus']
ritmo_cols = ['secao', 'sequencia', 'sequenciatotal', 'progresso_sequencia', 'corridasciclo', 'tempociclo']
panela_cols = ['panela', 'vidapanela']

dicionario_blocos = {
    'Química': quimica_cols,
    'Ritmo (Dinâmica)': ritmo_cols,
    'Panela': panela_cols,
}

# Mantemos a mesma divisão de dados para todos os blocos para ser uma comparação justa
idx_treino, idx_teste = train_test_split(df.index, test_size=0.2, random_state=42)
y_treino, y_teste = y.loc[idx_treino], y.loc[idx_teste]

# --- 3. TREINAMENTO E AVALIAÇÃO INDEPENDENTE ---
resultados_mae = {}

for nome_bloco, variaveis in dicionario_blocos.items():
    # Filtrar o dataset apenas com as colunas do respectivo bloco
    X_treino_bloco = df.loc[idx_treino, variaveis]
    X_teste_bloco = df.loc[idx_teste, variaveis]
    
    # Criar e treinar o XGBoost
    modelo_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    modelo_xgb.fit(X_treino_bloco, y_treino)
    
    # Prever e avaliar
    y_pred_bloco = modelo_xgb.predict(X_teste_bloco)
    mae_bloco = mean_absolute_error(y_teste, y_pred_bloco)
    resultados_mae[nome_bloco] = mae_bloco

# --- 4. EXIBIÇÃO DOS RESULTADOS ---
print("=== ERRO MÉDIO ABSOLUTO (MAE) POR BLOCO ISOLADO ===")
for bloco, mae in sorted(resultados_mae.items(), key=lambda item: item[1]):
    print(f"Bloco {bloco.ljust(18)}: {mae:.2f}°C")

# --- 5. GRÁFICO DE COMPARAÇÃO ---
plt.figure(figsize=(10, 6))
blocos_nomes = list(resultados_mae.keys())
maes_valores = list(resultados_mae.values())

bars = plt.bar(blocos_nomes, maes_valores, color=['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Erro por Bloco - XGBoost')
plt.ylabel('MAE (°C)')
plt.xlabel('Blocos de variáveis')

# Adicionando o valor do MAE em cima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()