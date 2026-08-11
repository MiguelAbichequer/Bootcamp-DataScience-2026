import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Importando as bibliotecas avançadas
import lightgbm as lgb
import xgboost as xgb

# --- 1. LEITURA E PRÉ-PROCESSAMENTO ---
df = pd.read_csv(r'Bootcamp-DataScience-2026\fundo_amanha_dados_mock_2026.csv', sep=',', encoding='latin-1')

df = df.drop_duplicates().dropna()
df = df[df['temperaturamediareal'] >= 1000]
df = df[df['sugestaomodelolegado'] >= 1000]

# Engenharias de features base
df['c_med'] = (df['c_min'] + df['c_max']) / 2
df['c_diff'] = df['c_max'] - df['c_min']
df['progresso_sequencia'] = df['sequencia'] / df['sequenciatotal']

df = pd.get_dummies(df, columns=['qualidade'], prefix='qualidade')

# Isolando variáveis de target e legado antes de dropar
y = df['temperaturasaidafp']
legado = df['sugestaomodelolegado']

# Removendo colunas desnecessárias para as features (X)
X = df.drop(['acoatual', 'c_min', 'c_max', 'corrida', 'sugestaomodelolegado', 'temperaturasaidafp', 'temperaturamediareal'], axis=1)

# Divisão de treino e teste (garantindo que o legado seja dividido nos mesmos índices)
X_treino, X_teste, y_treino, y_teste, legado_treino, legado_teste = train_test_split(
    X, y, legado, test_size=0.2, random_state=42
)

# --- 2. TREINAMENTO DOS MODELOS ---

# Modelo 1: Árvore de Decisão
tree_model = DecisionTreeRegressor(max_depth=X.shape[1]*5, random_state=42)
tree_model.fit(X_treino, y_treino)
y_pred_arvore = tree_model.predict(X_teste)

# Modelo 2: LightGBM
lgbm_model = lgb.LGBMRegressor(random_state=42)
lgbm_model.fit(X_treino, y_treino)
y_pred_lgbm = lgbm_model.predict(X_teste)

# Modelo 3: XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_treino, y_treino)
y_pred_xgb = xgb_model.predict(X_teste)

# Modelo 4: Legado (já temos as previsões em legado_teste)
y_pred_legado = legado_teste

# --- 3. CÁLCULO DOS ERROS (MAE) ---
mae_legado = mean_absolute_error(y_teste, y_pred_legado)
mae_arvore = mean_absolute_error(y_teste, y_pred_arvore)
mae_lgbm = mean_absolute_error(y_teste, y_pred_lgbm)
mae_xgb = mean_absolute_error(y_teste, y_pred_xgb)

resultados = {
    'Modelo Legado': mae_legado,
    'Árvore de Decisão': mae_arvore,
    'LightGBM': mae_lgbm,
    'XGBoost': mae_xgb
}

# Exibindo no console
print("=== COMPARAÇÃO DE PERFORMANCE DOS MODELOS (MAE) ===")
for nome, mae in resultados.items():
    print(f"{nome.ljust(20)}: {mae:.4f} °C")

# --- 4. VISUALIZAÇÃO DOS RESULTADOS ---
plt.figure(figsize=(10, 6))

modelos_nomes = list(resultados.keys())
maes_valores = list(resultados.values())

# Cores para destacar os melhores modelos (mais escuro/frio para os menores erros)
cores = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']

bars = plt.bar(modelos_nomes, maes_valores, color=cores, edgecolor='black', alpha=0.8)

plt.title('Comparação do Erro Médio Absoluto (MAE) entre Modelos', fontsize=14, fontweight='bold')
plt.ylabel('MAE (°C) - Quanto menor, melhor', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando os valores em cima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.2f} °C', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()


# --- 5. FEATURE IMPORTANCE (IMPORTÂNCIA DAS VARIÁVEIS) ---

# Quantidade de features para exibir no gráfico (Top 10)
top_n = 10
features = X.columns

# Função auxiliar para pegar as top N features de um modelo
def get_top_features(importances, feature_names, top_n):
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    return top_importances, top_features

# Extraindo a importância de cada modelo
imp_tree, feat_tree = get_top_features(tree_model.feature_importances_, features, top_n)
imp_lgbm, feat_lgbm = get_top_features(lgbm_model.feature_importances_, features, top_n)
imp_xgb, feat_xgb = get_top_features(xgb_model.feature_importances_, features, top_n)

# Criando a figura para plotar
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Top {top_n} Variáveis Mais Importantes por Modelo', fontsize=16, fontweight='bold')

# 1. Gráfico Árvore de Decisão
axes[0].barh(range(len(feat_tree)), imp_tree, align='center', color='#ff7f0e', edgecolor='black')
axes[0].set_yticks(range(len(feat_tree)))
axes[0].set_yticklabels(feat_tree)
axes[0].invert_yaxis() # Para a mais importante ficar no topo
axes[0].set_title('Árvore de Decisão')
axes[0].set_xlabel('Importância Relativa')
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

# 2. Gráfico LightGBM
axes[1].barh(range(len(feat_lgbm)), imp_lgbm, align='center', color='#1f77b4', edgecolor='black')
axes[1].set_yticks(range(len(feat_lgbm)))
axes[1].set_yticklabels(feat_lgbm)
axes[1].invert_yaxis()
axes[1].set_title('LightGBM')
axes[1].set_xlabel('Importância Relativa')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

# 3. Gráfico XGBoost
axes[2].barh(range(len(feat_xgb)), imp_xgb, align='center', color='#2ca02c', edgecolor='black')
axes[2].set_yticks(range(len(feat_xgb)))
axes[2].set_yticklabels(feat_xgb)
axes[2].invert_yaxis()
axes[2].set_title('XGBoost')
axes[2].set_xlabel('Importância Relativa')
axes[2].grid(axis='x', linestyle='--', alpha=0.7)

# Ajuste de layout para não cortar os nomes das variáveis
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()