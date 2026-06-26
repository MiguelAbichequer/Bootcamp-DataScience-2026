import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Importando o LightGBM
import lightgbm as lgb

# --- BLOCO LEITURA E PRÉ-PROCESSAMENTO ---
df = pd.read_csv(r'Bootcamp-DataScience-2026\fundo_amanha_dados_mock_2026.csv', sep=',', encoding='latin-1')

# df = df.drop_duplicates()
df = df.dropna()
df = df[df['temperaturamediareal'] >= 1000]
df = df[df['sugestaomodelolegado']>= 1000]

# BLOCO QUÍMICA:
df['c_med'] = (df['c_min'] + df['c_max'])/2
df['c_diff'] = df['c_max'] - df['c_min']
df = df.drop(['c_min', 'c_max'], axis=1)

df = df.drop(['panela'], axis=1)

# BLOCO DINÂMICA:
df['progresso_sequencia'] = df['sequencia'] / df['sequenciatotal']
df = pd.get_dummies(df, columns=['qualidade'], prefix='qualidade') 
df = df.drop(['acoatual'], axis=1)

# BLOCO PREPARAÇÃO DE DADOS
X = df.drop(['sugestaomodelolegado', 'temperaturasaidafp', 'temperaturamediareal'], axis=1) 
y = df['temperaturasaidafp'] 

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

# TREINANDO OS MODELOS

# Árvore de Regressão Original
tree = DecisionTreeRegressor(max_depth=X.shape[1]*5, random_state=42)
tree.fit(X_treino, y_treino)
y_pred_arvore = tree.predict(X_teste)

# 2. LightGBM
# Instanciamos o modelo LGBMRegressor. 
lgbm = lgb.LGBMRegressor(random_state=42)
lgbm.fit(X_treino, y_treino)
y_pred_lgbm = lgbm.predict(X_teste)

# 3. Modelo Legado
y_pred_legado = df.loc[X_teste.index, 'sugestaomodelolegado']

# --- AVALIAÇÃO ---
mae_legado = mean_absolute_error(y_teste, y_pred_legado)
mae_arvore = mean_absolute_error(y_teste, y_pred_arvore)
mae_lgbm = mean_absolute_error(y_teste, y_pred_lgbm)

print(f"MAE do Modelo Legado: {mae_legado:.4f}")
print(f"MAE da Árvore de Regressão: {mae_arvore:.4f}")
print(f"MAE do LightGBM: {mae_lgbm:.4f}\n")

if mae_lgbm < mae_arvore:
    print(f"Vitória dupla! O LightGBM reduziu o erro em {mae_legado - mae_lgbm:.2f}°C em relação ao legado, e {mae_arvore - mae_lgbm:.2f}°C em relação à árvore!")
else:
    print("Precisamos tunar os hiperparâmetros do LightGBM.")

# --- GRÁFICOS PARA COMPARAÇÃO ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Legado
axes[0].scatter(y_teste, y_pred_legado, alpha=0.5, color='salmon')
axes[0].set_title(f'Modelo Legado\nMAE: {mae_legado:.2f}')
axes[0].set_xlabel('Valores Reais')
axes[0].set_ylabel('Previsões')
axes[0].plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'k--', lw=2)

# Gráfico 2: Árvore
axes[1].scatter(y_teste, y_pred_arvore, alpha=0.5, color='mediumseagreen')
axes[1].set_title(f'Árvore de Decisão\nMAE: {mae_arvore:.2f}')
axes[1].set_xlabel('Valores Reais')
axes[1].plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'k--', lw=2)

# Gráfico 3: LightGBM
axes[2].scatter(y_teste, y_pred_lgbm, alpha=0.5, color='cornflowerblue')
axes[2].set_title(f'LightGBM\nMAE: {mae_lgbm:.2f}')
axes[2].set_xlabel('Valores Reais')
axes[2].plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'k--', lw=2)

plt.tight_layout()
plt.show()

# ... [todo o seu código existente até o plt.show()] ...

# --- BLOCO DE EXPORTAÇÃO DOS MODELOS ---

# Salvando a Árvore de Decisão
joblib.dump(tree, 'meu_modelo_arvore.pkl')

# Salvando o LightGBM
joblib.dump(lgbm, 'meu_modelo_lightgbm.pkl')

print("\nSucesso! Modelos (Árvore e LightGBM) exportados no formato .pkl usando Joblib!")