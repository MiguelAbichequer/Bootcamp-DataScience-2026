# Bootcamp 2026 Fundo do Amanhã

Desafio técnico do trainamento em ciência de dados oferecido pelo Fundo do Amanhã nas dependências do Instituo Caldeira.

## Descrição do desafio

Um conjunto de dados de uma empresa de lingotamento de metais dividos em:

Identificação e ciclo
"corrida": "Identificação única da corrida de aço no processo.",
"panela": "Identificação da panela utilizada na corrida.",
"vidapanela": "Vida acumulada da panela (quantas corridas ela já operou).",
"sequencia": "Posição da corrida dentro da sequência.",
"sequenciatotal": "Número total de corridas previstas para a sequência.",
"corridasciclo": "Número de corridas que a panela fez no ciclo"
Composição química mínima/máxima
"al_min": "Teor mínimo de alumínio especificado para a corrida.",
"c_min": "Teor mínimo de carbono especificado para a corrida.",
"c_max": "Teor máximo de carbono permitido na corrida.",
"n_min": "Teor mínimo de nitrogênio especificado.",
"s_min": "Teor mínimo de enxofre especificado.",
Qualidade e seção do aço
"acoatual": "Tipo/classe de aço produzido na corrida.",
"qualidade": "Qualidade do aço definida pelo pedido do cliente.",
"secao": "Seção transversal do produto no lingotamento (ex.: tarugo).",
Variáveis de processo
"temperaturaliquidus": "Temperatura de fusão (quando entra em estado
líquido).",
"temperaturasaidafp": "Temperatura medida na saída do Forno Panela
(FP).",
"tempociclo": "Tempo total da panela no ciclo.",
"sugestaomodelolegado": "Temperatura sugerida pelo modelo legado
(previsão histórica utilizada como referência)."
"temperaturamediareal": "Temperatura média real medida durante o
lingotamento.",
"temperaturaobjetivada": "Temperatura alvo definida para a corrida no
lingotamento(o valor a ser acertado)."

# Objetivos

Avaliar desempenho, relevância de dados, modelo utilizado.
Enriquecimento dos dados: dados podem ser “criados” e que
fazem sentido para o processo.
Proposta de um novo modelo: Treinar um ou mais modelos (ex: regressão
linear, XGBoost, redes neurais, etc). Avaliar se esse novo modelo supera o
legado em termos de métrica de modelo e processo.



