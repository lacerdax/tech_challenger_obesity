Perfeito 👍
Segue **o arquivo README.md pronto**, exatamente no formato para **copiar e colar** no seu repositório.

> 👉 Dica rápida: copie **tudo**, crie um arquivo chamado `README.md` na raiz do projeto e cole.

---

````markdown
# Tech Challenge – Fase 04

## Sistema Preditivo de Obesidade (Machine Learning + Streamlit)

Este projeto foi desenvolvido como parte do **Tech Challenge – Fase 04**, cujo objetivo é aplicar os conhecimentos de Ciência de Dados e Machine Learning para construir um **modelo preditivo capaz de auxiliar a equipe médica na identificação do nível de obesidade de pacientes**, a partir de dados físicos, comportamentais e de hábitos de vida.

---

## 🎯 Objetivo do Projeto

Desenvolver uma **pipeline completa de Machine Learning**, contemplando:

- Feature engineering
- Treinamento e avaliação de modelo preditivo
- Modelo com assertividade superior a **75%**
- Deploy do modelo em uma **aplicação preditiva utilizando Streamlit**

O sistema tem caráter **educacional e de apoio à decisão**, não substituindo avaliação clínica profissional.

---

## 📊 Base de Dados

Arquivo utilizado: **`Obesity.csv`**

A base contém informações demográficas, físicas e comportamentais, além da variável alvo:

- **Variável alvo:** `Obesity` (classificação multiclasse)
- Classes:
  - Insufficient_Weight
  - Normal_Weight
  - Overweight_Level_I
  - Overweight_Level_II
  - Obesity_Type_I
  - Obesity_Type_II
  - Obesity_Type_III

A base apresenta boa distribuição entre as classes, o que favorece o treinamento de modelos robustos.

---

## 🧠 Feature Engineering

A etapa de feature engineering foi implementada seguindo boas práticas de Machine Learning:

### 🔹 Variáveis Numéricas

- Age
- Height
- Weight
- FCVC
- NCP
- CH2O
- FAF
- TUE

**Tratamento aplicado:**

- Padronização com `StandardScaler`, garantindo que variáveis em diferentes escalas não influenciem indevidamente o modelo.

### 🔹 Variáveis Categóricas

- Gender
- family_history
- FAVC
- CAEC
- SMOKE
- SCC
- CALC
- MTRANS

**Tratamento aplicado:**

- Codificação One-Hot (`OneHotEncoder`) com tratamento de categorias desconhecidas (`handle_unknown="ignore"`).

Todo o pré-processamento foi encapsulado em um **`ColumnTransformer`**, garantindo consistência entre treino e inferência em produção e prevenindo _data leakage_.

---

## ⚙️ Pipeline de Machine Learning

A pipeline foi construída utilizando o `Pipeline` do scikit-learn, integrando:

1. Feature engineering
2. Modelo de classificação

Modelos avaliados:

- Logistic Regression Multinomial (baseline)
- RandomForestClassifier (modelo final)

O Random Forest, após ajuste de hiperparâmetros via `GridSearchCV`, apresentou o melhor desempenho.

---

## 📈 Avaliação do Modelo

- Estratégia de validação: **Train/Test Split estratificado (80/20)**
- Métrica principal: **Accuracy**
- Métricas adicionais:
  - Classification Report
  - Matriz de confusão

🎯 **Resultado:**  
O modelo final atingiu **assertividade superior a 75%**, atendendo plenamente aos requisitos do desafio.

---

## 🚀 Deploy da Aplicação (Streamlit)

O modelo treinado foi disponibilizado por meio de uma **aplicação web interativa desenvolvida com Streamlit**, permitindo:

- Inserção manual dos dados do paciente
- Predição do nível de obesidade
- Visualização das probabilidades por classe (quando aplicável)

A aplicação utiliza exatamente o mesmo pipeline de pré-processamento empregado no treinamento, garantindo confiabilidade das previsões.

---

## 🗂️ Estrutura do Repositório

```text
tech-challenge-obesity/
│
├── data/
│   └── Obesity.csv
│
├── models/
│   └── obesity_model.joblib
│
├── src/
│   ├── train.py        # Treinamento e avaliação do modelo
│   └── app.py          # Aplicação Streamlit
│
├── requirements.txt
└── README.md
```
````

---

## ▶️ Como Executar o Projeto Localmente

### 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Treinar o modelo

```bash
python src/train.py
```

### 3️⃣ Executar a aplicação Streamlit

```bash
streamlit run src/app.py
```

---

## 📌 Observações Finais

- O projeto foi desenvolvido com foco em **boas práticas de Ciência de Dados**, utilizando pipeline reprodutível e evitando vazamento de dados.
- O sistema é um **apoio à tomada de decisão**, não devendo ser utilizado como diagnóstico médico definitivo.
- O modelo pode ser estendido futuramente com técnicas de interpretabilidade e novas variáveis.

---

## 👨‍💻 Autor

Projeto desenvolvido como parte do **Tech Challenge – Fase 04**
Pós-Graduação em Análise de Dados – FIAP

```
Lucas Lacerda

```
