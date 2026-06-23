# 💳 FraudVision AI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blueviolet?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 AI-Powered Credit Card Fraud Detection System

**FraudVision AI** is an end-to-end Machine Learning application that detects fraudulent credit card transactions using a **Random Forest Classifier**. The project demonstrates the complete ML pipeline, including data preprocessing, feature scaling, handling class imbalance with **SMOTE**, model evaluation, and deployment through an interactive **Streamlit** dashboard.

---

# ✨ Features

- 📊 Interactive Streamlit Dashboard
- 🤖 Random Forest Classifier
- ⚖️ SMOTE for Imbalanced Dataset
- 📈 Accuracy, Precision, Recall, F1-Score & ROC-AUC
- 📉 Transaction Distribution Analysis
- 🧩 Interactive Confusion Matrix Heatmap
- ⭐ Top 10 Feature Importance
- 📈 ROC Curve Visualization
- 🔍 Manual Transaction Prediction
- 📂 Batch Prediction using CSV Upload
- 📥 Download Prediction Results
- 🎨 Modern Dashboard UI with Plotly Charts

---

# 📂 Project Structure

```text
Syntecxhub_Credit_Card_Fraud_Detection/
│
├── dataset/
│   └── creditcard.csv        # Download separately
│
├── images/
│   ├── dashboard.png
│   ├── dataset.png
│   ├── prediction.png
│   └── batch_prediction.png
│
├── app.py
├── train_model.py
├── utils.py
├── style.css
├── requirements.txt
├── scaler.pkl
├── README.md
└── .gitignore
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Machine Learning | Scikit-Learn |
| Algorithm | Random Forest |
| Data Processing | Pandas, NumPy |
| Imbalanced Learning | SMOTE |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Model Storage | Joblib |

---

# 📊 Machine Learning Workflow

```
Credit Card Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
SMOTE (Handle Class Imbalance)
        │
        ▼
Random Forest Training
        │
        ▼
Model Evaluation
        │
        ▼
Save Model
        │
        ▼
Streamlit Deployment
```

---

# 📈 Model Evaluation

The trained model is evaluated using multiple performance metrics:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Feature Importance
- ✅ ROC Curve

---

# 📁 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

📥 Download the dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file inside:

```text
dataset/
    creditcard.csv
```

---

# 🤖 Generate the Trained Model

The trained **model.pkl** file is not included because it exceeds GitHub's file size limit.

Generate it locally by running:

```bash
python train_model.py
```

This automatically creates:

- `model.pkl`
- `scaler.pkl`

---

# ▶️ Installation

### Clone the repository

```bash
git clone https://github.com/aryansinghnaman/Syntecxhub_creditcard_fraud_detection.git
```

### Move into the project directory

```bash
cd Syntecxhub_creditcard_fraud_detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate the trained model

```bash
python train_model.py
```

### Launch the application

```bash
streamlit run app.py
```

---

# 📸 Application Preview

## 📊 Dashboard

![Dashboard](images/dashboard.png)

---

## 📁 Dataset Analysis

![Dataset](images/dataset.png)

---

## 🔍 Manual Prediction

![Prediction](images/prediction.png)

---

## 📂 Batch CSV Prediction

![Batch Prediction](images/batch_prediction.png)

---

# 💼 Real-World Workflow

```
User Input
      │
      ▼
Feature Scaling
      │
      ▼
Random Forest Model
      │
      ▼
Fraud Probability
      │
      ▼
Prediction
      │
      ▼
Dashboard Visualization
```

---

# 💡 Future Improvements

- XGBoost Implementation
- LightGBM Comparison
- Hyperparameter Optimization
- SHAP Explainability
- REST API Integration
- Docker Deployment
- Cloud Deployment
- Real-Time Fraud Detection API

---

# 👨‍💻 Developer

## Aryan Singh

**Machine Learning Intern @ Syntecxhub**

### Connect with me

- 🔗 GitHub: https://github.com/aryansinghnaman
- 💼 LinkedIn: https://linkedin.com/in/aryansingh9557

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---

## 📜 License

This project is created for educational and portfolio purposes.
