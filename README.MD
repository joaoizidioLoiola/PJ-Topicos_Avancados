# Análise Preditiva de Doenças Cardiovasculares

## 📋 Descrição
Projeto de machine learning para predição de doenças cardiovasculares utilizando redes neurais e monitoramento de consumo energético.

## 🔧 Requisitos

### Instalação das dependências
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow codecarbon
```

## 📦 Estrutura do Projeto
```
projeto/
├── data/
│   └── Cardiovascular_Disease_Dataset.csv
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
└── notebooks/
 └── analysis.ipynb
```

## 🚀 Como Usar

### 1. Preparação dos Dados
```python
# Carregar e preparar dados
data = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
X = data.drop(['patientid', 'target'], axis=1)
y = data['target']
```

### 2. Normalização
```python
# Normalizar dados
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Treinamento do Modelo
```python
# Configurar e treinar modelo
model = keras.Sequential([
 keras.Input(shape=(12,)),
 layers.Dense(100, activation='relu'),
 layers.Dropout(0.5),
 layers.Dense(100, activation='relu'),
 layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
     loss='binary_crossentropy', 
     metrics=['accuracy'])
```

## 📊 Visualizações
- Matriz de Correlação
- Distribuição de Classes
- Curva ROC
- Matriz de Confusão

## 📈 Resultados
- Acurácia: ~96%
- Precisão por classe
- Consumo energético monitorado

## 🔬 Métricas de Avaliação
```python
# Avaliação do modelo
print(classification_report(y_test, y_pred))
```

## 🌱 Monitoramento Ambiental
```python
# Monitorar consumo energético
tracker = EmissionsTracker()
tracker.start()
# ... treinamento ...
tracker.stop()
```

## 📝 Licença
Este projeto está sob a licença MIT.

## 📚 Referências
- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)