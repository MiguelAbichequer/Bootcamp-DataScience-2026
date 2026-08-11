# gerar previsões da árvore para as corridas de teste
y_pred_arvore = tree.predict(X_teste)

# previsões do modelo legado para as corridas de teste
y_pred_legado = df.loc[X_teste.index, 'sugestaomodelolegado']

# ==========================================
# MELHORIA DOS GRÁFICOS
# ==========================================

# Definir os limites do gráfico para traçar a reta ideal (onde Real = Previsto)
min_val = min(y_teste.min(), y_pred_arvore.min(), y_pred_legado.min())
max_val = max(y_teste.max(), y_pred_arvore.max(), y_pred_legado.max())

# Criar uma figura com 3 gráficos lado a lado
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Comparação de Desempenho dos Modelos (Valores Reais vs Previsões)', fontsize=16)

# Gráfico 1: Apenas Árvore de Regressão
axes[0].scatter(y_teste, y_pred_arvore, alpha=0.5, color='royalblue', label='Previsões da Árvore')
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[0].set_title('1. Árvore de Regressão', fontsize=14)
axes[0].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[0].set_ylabel('Previsões (°C)', fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Gráfico 2: Apenas Modelo Legado
axes[1].scatter(y_teste, y_pred_legado, alpha=0.5, color='forestgreen', label='Previsões do Modelo Legado')
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[1].set_title('2. Modelo Legado', fontsize=14)
axes[1].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Gráfico 3: Ambos Juntos
axes[2].scatter(y_teste, y_pred_arvore, alpha=0.5, color='royalblue', label='Árvore')
axes[2].scatter(y_teste, y_pred_legado, alpha=0.5, color='forestgreen', marker='x', label='Legado')
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
axes[2].set_title('3. Sobreposição dos Modelos', fontsize=14)
axes[2].set_xlabel('Valores Reais (°C)', fontsize=12)
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==========================================
# MÉTRICAS E RESULTADO
# ==========================================

mae_arvore = mean_absolute_error(y_teste, y_pred_arvore)
mae_legado = mean_absolute_error(y_teste, y_pred_legado)

print("-" * 50)
print(f"MAE da Árvore de Regressão: {mae_arvore:.2f} °C")
print(f"MAE do Modelo Legado: {mae_legado:.2f} °C")
print("-" * 50)

if mae_arvore < mae_legado:
    print(f"Vitória! A árvore reduziu o erro médio em {(mae_legado - mae_arvore):.2f}°C!")
else:
    print("O modelo legado ainda está ganhando. Precisamos ajustar os hiperparâmetros (como a profundidade) da árvore.")