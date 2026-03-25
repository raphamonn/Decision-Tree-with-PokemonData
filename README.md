# 📊 Classificação de Pokémons Lendários com Machine Learning

## 📌 Visão Geral
Este projeto tem como objetivo construir um modelo de **Machine Learning** capaz de classificar se um Pokémon é **lendário ou não** com base em suas características (features), como peso, altura e taxa de captura (*catch rate*).

A abordagem utiliza algoritmos de **classificação supervisionada**, com foco em **Decision Trees**, além de técnicas de **redução de dimensionalidade (PCA)** e **validação cruzada**.

---

## 🎯 Problema
Dado um conjunto de atributos de um Pokémon, é possível prever se ele é lendário?

Esse problema é interessante pois:
- Envolve **dados desbalanceados** (poucos lendários vs muitos comuns)
- Permite explorar **interpretabilidade do modelo**
- Simula cenários reais como detecção de padrões raros

---

## 🎯 Objetivos

- Construir um modelo de classificação para prever se um Pokémon é lendário
- Analisar a **correlação entre as features**
- Avaliar o desempenho com métricas como **acurácia**
- Interpretar a importância das variáveis
- Comparar:
  - Modelo sem PCA
  - Modelo com PCA
- Visualizar os dados em 2D após redução de dimensionalidade

---

## 🗂️ Dataset

- Fonte: Kaggle  
- Nome: *Pokémon for Data Mining and Machine Learning*  
- Autor: Asier López Zorrilla  

🔗 https://www.kaggle.com/datasets/alopez247/pokemon

### 📊 Características:
- 721 Pokémons
- 21 atributos
- Classes:
  - 46 lendários
  - 675 não lendários

⚠️ Dataset **desbalanceado**

---

## 🛠️ Tecnologias e Bibliotecas

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Graphviz

---

## 🔄 Pipeline do Projeto

### 1. Carregamento dos Dados
- Importação do dataset
- Visualização inicial (`head`, `shape`)

### 2. Pré-processamento
- Tratamento de valores nulos:
  - `Pr_Male` preenchido com 0.5
- Remoção de colunas irrelevantes:
  - `Egg_Group_2`
  - `Type_2`

### 3. Análise Exploratória
- Matriz de correlação
- Identificação de relações importantes:
  - Correlação negativa entre `isLegendary` e `catch_rate`

### 4. Separação dos Dados
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

### 5. Modelagem
🔹 Decision Tree
- Treinamento com dados originais
- Análise de importância das variáveis
🔹 Decision Tree + PCA
- Padronização dos dados
- Redução de dimensionalidade com PCA
- Treinamento com dados transformados

### 6. Validação
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
```
### 7. Avaliação
- Métrica principal: Acurácia
- Comparação entre modelos

### 8. Visualização
- Heatmap de correlação
- PCA em 2D
- Estrutura da árvore de decisão

## 📊 Resultados
- O modelo identificou padrões relevantes entre as variáveis
- catch_rate se destacou como feature importante
- PCA ajudou na visualização dos dados
- Decision Trees forneceram boa interpretabilidade

## ⚖️ Limitações
- Dataset desbalanceado
- Uso de poucas métricas (poderia incluir Precision, Recall, F1-score)
- Modelos mais robustos não foram explorados

## 🚀 Melhorias Futuras
- Testar modelos como:
  - Random Forest
  - XGBoost
- Aplicar balanceamento de classes (SMOTE, undersampling)
- Utilizar GridSearchCV ou RandomizedSearchCV
- Feature Engineering
- Deploy do modelo

