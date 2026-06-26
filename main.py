import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r'Bootcamp-DataScience-2026\fundo_amanha_dados_mock_2026.csv', sep=',', encoding='latin-1')

df = df.drop_duplicates()
df = df.dropna()
df = df[df['temperaturamediareal'] >= 1000]
df = df[df['sugestaomodelolegado']>= 1000]


# BLOCO QUÍMICA:

df['c_med'] = (df['c_min'] + df['c_max'])/2

#df['c_medh'] = df

df['c_diff'] = df['c_max'] - df['c_min']

df = df.drop(['c_min', 'c_max'], axis=1)

df = df.drop(['panela'], axis=1)

df = df.drop(['corrida'], axis=1)

# BLOCO DINÂMICA:


df['progresso_sequencia'] = df['sequencia'] / df['sequenciatotal'] # porcentagem de conclusão da sequência

# traduzir dados texto para valores 

# mapear qualidades para valores booleanos:

df = pd.get_dummies(df, columns=['qualidade'], prefix='qualidade') # one-hot encoding para a coluna 'qualidade'
# df = df.drop(['qualidade'], axis=1)

# mapear acoatual:

#df = pd.get_dummies(df, columns=['acoatual'], prefix='acoatual')

df = df.drop(['acoatual'], axis=1)



# Retirar dados reposta para o treinamento do modelo


X = df.drop(['sugestaomodelolegado', 'temperaturasaidafp', 'temperaturamediareal'], axis=1) # 
y = df['temperaturasaidafp'] # esta é a resposta correta 


print(X.tail())


# treinar árvore de regressão

# divisão dos dados:

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

# criar e treinar árvore

tree = DecisionTreeRegressor(max_depth=X.shape[1]).fit(X_treino, y_treino)

# comparar modelos

# gerar previsões da árvore para as corridas de teste
y_pred_arvore = tree.predict(X_teste)

# previsões do modelo legado para as corridas de teste
y_pred_legado = df.loc[X_teste.index, 'sugestaomodelolegado']

# ==========================================
# GRÁFICOS MELHORADOS: COMPARAÇÃO DE MODELOS
# ==========================================

# 1. Definir os limites dos eixos para que todos os gráficos fiquem na mesma escala
min_val = min(y_teste.min(), y_pred_arvore.min(), y_pred_legado.min())
max_val = max(y_teste.max(), y_pred_arvore.max(), y_pred_legado.max())

# 2. Criar uma figura com 3 gráficos lado a lado (1 linha, 3 colunas)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Comparação de Desempenho: Árvore de Regressão vs Modelo Legado', fontsize=16)

# --- Gráfico 1: Árvore de Regressão ---
axes[0].scatter(y_teste, y_pred_arvore, alpha=0.5, color='mediumseagreen', label='Previsões Árvore')
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[0].set_title('1. Árvore de Regressão', fontsize=14)
axes[0].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[0].set_ylabel('Previsões (°C)', fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Gráfico 2: Modelo Legado ---
axes[1].scatter(y_teste, y_pred_legado, alpha=0.5, color='salmon', label='Previsões Legado')
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[1].set_title('2. Modelo Legado', fontsize=14)
axes[1].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Gráfico 3: Sobreposição ---
axes[2].scatter(y_teste, y_pred_arvore, alpha=0.5, color='mediumseagreen', label='Árvore de Regressão')
axes[2].scatter(y_teste, y_pred_legado, alpha=0.5, color='salmon', marker='x', label='Modelo Legado')
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[2].set_title('3. Sobreposição dos Modelos', fontsize=14)
axes[2].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

# Ajustar o layout para não cortar os textos e mostrar a figura
plt.tight_layout()
plt.show()

mae_arvore = mean_absolute_error(y_teste, y_pred_arvore)
mae_legado = mean_absolute_error(y_teste, y_pred_legado)

print(f"MAE da Árvore de Regressão: {mae_arvore}")
print(f"MAE do Modelo Legado: {mae_legado}")


if mae_arvore < mae_legado:
    print(f"Vitória! A árvore reduziu o erro em {mae_legado - mae_arvore:.2f}°C!")
else:
    print("O modelo legado ainda está ganhando. Precisamos ajustar a profundidade da árvore.")


# --- BLOCO DE EXPORTAÇÃO ---
# Como estamos usando Scikit-Learn (DecisionTreeRegressor), o Joblib é a ferramenta correta.

joblib.dump(tree, 'meu_modelo_arvore.pkl')
print("Modelo de Árvore de Regressão salvo com sucesso no formato .pkl usando Joblib!")