# 💳 FraudVision AI
### AI-Powered Credit Card Fraud Detection System

FraudVision AI is an end-to-end Machine Learning project that detects fraudulent credit card transactions using a **Random Forest Classifier**. The project includes data preprocessing, handling class imbalance with **SMOTE**, model evaluation, and an interactive **Streamlit** dashboard for real-time and batch predictions.

---

## 🚀 Features

- 📊 Interactive Streamlit Dashboard
- 🤖 Credit Card Fraud Detection using Random Forest
- ⚖️ SMOTE for handling imbalanced data
- 📈 Model Performance Metrics
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- 📉 Transaction Distribution Visualization
- 🧩 Confusion Matrix Heatmap
- ⭐ Top 10 Feature Importance
- 📈 ROC Curve
- 🔍 Manual Transaction Prediction
- 📂 Batch Prediction using CSV Upload
- 📥 Download Prediction Results

---

## 📂 Project Structure

```text
Syntecxhub_Credit_Card_Fraud_Detection/
│
├── dataset/
│   └── creditcard.csv      # Download separately
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
├── README.md
├── scaler.pkl
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-Learn (SMOTE)
- Plotly
- Joblib

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Train-Test Split
4. Feature Scaling
5. Handle Class Imbalance using SMOTE
6. Train Random Forest Classifier
7. Evaluate Model Performance
8. Save Trained Model
9. Deploy using Streamlit

---

## 📁 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

Download the dataset from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file inside:

```text
dataset/
    creditcard.csv
```

---

## 🤖 Generate the Model

The trained `model.pkl` file is not included because of GitHub file size limitations.

Generate it locally by running:

```bash
python train_model.py
```

This will create:

- `model.pkl`
- `scaler.pkl`

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/aryansinghnaman/Syntecxhub_Credit_Card_Fraud_Detection.git
```

Move into the project folder:

```bash
cd Syntecxhub_Credit_Card_Fraud_Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training script (only if `model.pkl` is missing):

```bash
python train_model.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 📸 Application Preview

### Dashboard
_Add `images/dashboard.png`_

### Dataset
_Add `images/dataset.png`_

### Prediction
_Add `images/prediction.png`_

### Batch Prediction
_Add `images/batch_prediction.png`_

---

## 💡 Future Improvements

- XGBoost implementation
- LightGBM comparison
- Hyperparameter tuning
- Explainable AI (SHAP/LIME)
- Cloud deployment
- REST API integration

---

## 👨‍💻 Developer

**Aryan Singh**

Machine Learning Intern @ Syntecxhub

- GitHub: https://github.com/aryansinghnaman
- LinkedIn: https://linkedin.com/in/aryansingh9557

---

## ⭐ If you found this project useful, consider giving it a star!
