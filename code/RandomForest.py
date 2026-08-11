import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split
# Importando o Random Forest no lugar da Decision Tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. CARREGAMENTO DOS DADOS
df = pd.read_csv(r'Bootcamp-DataScience-2026\fundo_amanha_dados_mock_2026.csv', sep=',', encoding='latin-1')

df = df.drop_duplicates()
df = df.dropna()
df = df[df['temperaturamediareal'] >= 1000]
df = df[df['sugestaomodelolegado'] >= 1000]

# 2. BLOCO QUÍMICA
df['c_med'] = (df['c_min'] + df['c_max'])/2
df['c_diff'] = df['c_max'] - df['c_min']
df = df.drop(['c_min', 'c_max', 'panela'], axis=1)

# 3. BLOCO DINÂMICA
df['progresso_sequencia'] = df['sequencia'] / df['sequenciatotal']

# Transformar variáveis de texto (qualidade E acoatual) em colunas booleanas
df = pd.get_dummies(df, columns=['qualidade', 'acoatual'], prefix=['qualidade', 'acoatual'])

# REMOVER O ID DA CORRIDA (Impede que o modelo tente decorar as linhas)
if 'corrida' in df.columns:
    df = df.drop(['corrida'], axis=1)

# 4. SEPARAR VARIÁVEIS (X) E RESPOSTA (y)
X = df.drop(['sugestaomodelolegado', 'temperaturasaidafp', 'temperaturamediareal'], axis=1) 
y = df['temperaturasaidafp'] 

# 5. DIVISÃO DOS DADOS
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

# 6. CRIAR E TREINAR A FLORESTA ALEATÓRIA (Random Forest)
# n_estimators=100 significa que estamos criando 100 árvores de decisão e combinando-as!
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_treino, y_treino)

# 7. COMPARAR MODELOS
y_pred_rf = modelo_rf.predict(X_teste)
y_pred_legado = df.loc[X_teste.index, 'sugestaomodelolegado']

# 8. CÁLCULO DO ERRO (MAE)
mae_rf = mean_absolute_error(y_teste, y_pred_rf)
mae_legado = mean_absolute_error(y_teste, y_pred_legado)

print("-" * 50)
print(f"MAE do Modelo Legado: {mae_legado:.2f} °C")
print(f"MAE do Random Forest: {mae_rf:.2f} °C")
print("-" * 50)

if mae_rf < mae_legado:
    print(f"Vitória! O Random Forest reduziu o erro do modelo legado em {mae_legado - mae_rf:.2f}°C!")
else:
    print("O modelo legado ainda está ganhando.")

# ==========================================
# PLOTANDO OS GRÁFICOS MELHORADOS
# ==========================================
min_val = min(y_teste.min(), y_pred_rf.min(), y_pred_legado.min())
max_val = max(y_teste.max(), y_pred_rf.max(), y_pred_legado.max())

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Comparação de Desempenho dos Modelos (Valores Reais vs Previsões)', fontsize=16)

# Gráfico 1: Apenas Random Forest
axes[0].scatter(y_teste, y_pred_rf, alpha=0.5, color='royalblue', label='Previsões RF')
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[0].set_title('1. Random Forest', fontsize=14)
axes[0].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[0].set_ylabel('Previsões (°C)', fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Gráfico 2: Modelo Legado
axes[1].scatter(y_teste, y_pred_legado, alpha=0.5, color='forestgreen', label='Previsões Legado')
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[1].set_title('2. Modelo Legado', fontsize=14)
axes[1].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Gráfico 3: Sobreposição
axes[2].scatter(y_teste, y_pred_rf, alpha=0.5, color='royalblue', label='Random Forest')
axes[2].scatter(y_teste, y_pred_legado, alpha=0.5, color='forestgreen', marker='x', label='Legado')
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[2].set_title('3. Sobreposição dos Modelos', fontsize=14)
axes[2].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

joblib.dump(modelo_rf, 'meu_modelo_rf.pkl')

print("Sucesso! Modelo Random Forest exportado no formato .pkl usando Joblib!")