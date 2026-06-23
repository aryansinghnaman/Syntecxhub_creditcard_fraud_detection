import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset/creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# Features & Target
# ==============================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# SMOTE
# ==============================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE")
print(pd.Series(y_train).value_counts())

# ==============================
# Random Forest Model
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ==============================
# Evaluation
# ==============================

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==============================
# Save Model
# ==============================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")