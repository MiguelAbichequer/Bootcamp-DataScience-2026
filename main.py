import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

plt.scatter(y_teste, y_pred_arvore)
plt.xlabel('Valores Reais')
plt.ylabel('Previsões da Árvore de Regressão')
plt.show()

plt.scatter(y_teste, y_pred_legado)
plt.xlabel('Valores Reais')
plt.ylabel('Previsões do Modelo Legado')
plt.show()

mae_arvore = mean_absolute_error(y_teste, y_pred_arvore)
mae_legado = mean_absolute_error(y_teste, y_pred_legado)

print(f"MAE da Árvore de Regressão: {mae_arvore}")
print(f"MAE do Modelo Legado: {mae_legado}")


if mae_arvore < mae_legado:
    print(f"Vitória! A árvore reduziu o erro em {mae_legado - mae_arvore:.2f}°C!")
else:
    print("O modelo legado ainda está ganhando. Precisamos ajustar a profundidade da árvore.")


