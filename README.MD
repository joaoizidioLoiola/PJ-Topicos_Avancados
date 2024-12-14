### 🫀 Análise Preditiva de Doenças Cardiovasculares

Este projeto de **Machine Learning** aplica técnicas de **Aprendizado Supervisionado** e **Redes Neurais Profundas** para análise preditiva de doenças cardiovasculares. Inclui **visualizações avançadas** e **monitoramento de consumo energético**, com o objetivo de identificar padrões e prever a presença de doenças cardiovasculares.

---

## 📋 Funcionalidades

1. **Preparação e Análise dos Dados**:
   - Exploração inicial de dados para entender distribuições e correlações.
   - Normalização das variáveis para otimizar o desempenho dos modelos.

2. **Modelagem e Predição**:
   - Modelos tradicionais de Machine Learning como Regressão Logística.
   - Modelos avançados de Deep Learning usando TensorFlow/Keras.
   - Redução de dimensionalidade utilizando PCA.

3. **Visualizações Informativas**:
   - Histogramas, gráficos de barras, e matrizes de correlação para análise exploratória.
   - Curva ROC e Matriz de Confusão para avaliar o desempenho do modelo.

4. **Monitoramento Sustentável**:
   - Rastreamento do consumo energético e cálculo da pegada de carbono com a biblioteca **CodeCarbon**.

---

## 📦 Estrutura do Projeto

```plaintext
projeto/
├── data/
│   └── Cardiovascular_Disease_Dataset.csv  # Dataset principal
├── src/
│   ├── preprocessing.py  # Scripts de pré-processamento
│   ├── model.py          # Definição e treinamento dos modelos
│   └── visualization.py  # Geração de gráficos e análises visuais
└── notebooks/
    └── analysis.ipynb    # Notebook para análise exploratória

```

--- 

## 🔧 Requisitos

Antes de começar, certifique-se de ter o Python 3.7+ instalado. Instale as dependências do projeto com o comando abaixo:


```
bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow codecarbon
```

---
## 🚀 Como Usar
## 1️⃣ Carregar e Preparar os Dados
```
python import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```
## Carregar dados
```
data = pd.read_csv("data/Cardiovascular_Disease_Dataset.csv")
```

## Separar features e target
```
X = data.drop(['patientid', 'target'], axis=1)
y = data['target']
```
## Normalizar os dados
```
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

## 2️⃣ Treinamento do Modelo de Deep Learning
```
python from tensorflow.keras import layers, Sequential
```

## Configurar e treinar o modelo
```
model = Sequential([
    layers.Input(shape=(X_scaled.shape[1],)),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
```

## Treinamento
```
model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2)
```
## 3️⃣ Avaliar o Modelo
```
python from sklearn.metrics import classification_report
```

## Predição e relatório
```
y_pred = (model.predict(X_scaled) > 0.5).astype(int)
print(classification_report(y, y_pred))
```

--- 
## 📊 Visualizações

1. **Distribuições** 
     - Histogramas e gráficos de barras ajudam a explorar variáveis como idade e tipo de dor no peito.
2. **Matriz de Correlação**
     - Identifica relações entre as variáveis.
3. **Curva ROC** 
     - Avalia a sensibilidade e especificidade do modelo.
4. **Matriz de Confusão** 
     - Visualiza erros de classificação.
---
## 🌱 Monitoramento Ambiental
O projeto incorpora o monitoramento do consumo energético usando CodeCarbon:
```
python from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
```
## Treinamento ou processos de alto custo computacional

```
tracker.stop()
```
---
## 📈 Resultados
1. **Acurácia**
     - Regressão Logística: ~79%
2. **Rede Neural**
     - ~96%
3. **Principais Insights**
     - Colesterol elevado está altamente correlacionado com doenças em homens.
     - Frequência cardíaca máxima é um preditor chave.   
---
## 📝 Licença
Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
--- 
## 📚 Referências
Documentação TensorFlow
Documentação Scikit-learn
Documentação AWS SageMaker