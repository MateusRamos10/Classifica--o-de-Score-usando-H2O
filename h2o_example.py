# Imports
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# Inicialização do H2O
h2o.init()

# Passo 0 - Entender o desafio
# Passo 1 - Importar base de dados
# Passo 2 - Preparar base de dados
    # 2.1 Verificar dados do Dataset
    # 2.2 Verificar informações vazias
    # 2.3 Verificar colunas não numéricas
    # 2.4 Converter para colunas numéricas
# Passo 3 - Criar modelo de score - Bom, Médio, Ruim
    # 3.1 Selecionar as colunas de treino e de teste
# Passo 4 - Treinar os modelos
# Passo 5 - Verificar melhor modelo
    # 5.1 Melhorar performance do melhor modelo
# Passo 6 - Usar modelo em cenário real
# Passo 7 - Resultado

# Passo 1 - Importar base de dados
clientes = pd.read_csv("clientes.csv") # Importa base de dados

# Convertendo DataFrame Pandas para H2OFrame
clientes_h2o = h2o.H2OFrame(clientes)

# Passo 2 - Preparar base de dados
    # 2.1 Verificar dados do Dataset
print(clientes_h2o.shape) # Checando informações, linhas e colunas
print(clientes_h2o.types) # Tipos de dados

# Passo 3 - Criar modelo de score - Bom, Médio, Ruim
    # 3.1 Selecionar as colunas de treino e de teste
x = clientes_h2o.columns[:-2]
y = 'score_credito'

# Passo 4 - Treinar os modelos
aml = H2OAutoML(max_runtime_secs=300, max_models=10, seed=1)  # Tempo máximo de execução de 5 minutos (300 segundos)
aml.train(x=x, y=y, training_frame=clientes_h2o)

# Passo 5 - Verificar melhor modelo
lb = aml.leaderboard
print(lb)

# Passo 6 - Usar modelo em cenário real
novos_clientes = pd.read_csv("novos_clientes.csv")
novos_clientes_h2o = h2o.H2OFrame(novos_clientes)

# Realizando previsões
previsoes = aml.predict(novos_clientes_h2o)

# Convertendo resultados para DataFrame Pandas
previsoes_df = previsoes.as_data_frame()

# Passo 7 - Resultado
resultado = pd.DataFrame({'Codigo do Cliente': novos_clientes['id_cliente'], 'Previsao Cliente': previsoes_df['predict']})
print(resultado)