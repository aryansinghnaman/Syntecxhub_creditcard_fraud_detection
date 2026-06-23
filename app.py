from sklearn.metrics import roc_curve
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------
# CSS
# --------------------------------

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("dataset/creditcard.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------
# TITLE
# --------------------------------

st.title("💳 FraudVision AI")

st.markdown("""
### AI Powered Credit Card Fraud Detection Dashboard

Welcome to **FraudVision AI**, an intelligent fraud detection system built using **Machine Learning**.

This project uses:

- 🤖 Random Forest Classifier
- ⚖️ SMOTE for handling imbalanced data
- 📊 Plotly Interactive Charts
- 🚀 Streamlit Dashboard
""")

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------


st.sidebar.title("💳 FraudVision AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Dashboard",
        "Dataset",
        "Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("### Project Details")

st.sidebar.write("✔ Random Forest")
st.sidebar.write("✔ SMOTE")
st.sidebar.write("✔ Scikit-Learn")
st.sidebar.write("✔ Streamlit")
st.sidebar.write("✔ Plotly")
# =====================================================
# DASHBOARD
# =====================================================
# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.header("📊 Dashboard")

    # -----------------------
    # MODEL METRICS
    # -----------------------

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("🎯 Accuracy",f"{acc:.4f}")
    c2.metric("✅ Precision",f"{pre:.4f}")
    c3.metric("📈 Recall",f"{rec:.4f}")
    c4.metric("⚡ F1 Score",f"{f1:.4f}")
    c5.metric("📊 ROC-AUC",f"{auc:.4f}")

    st.divider()

    # -----------------------
    # DATASET SUMMARY
    # -----------------------

    total = len(df)
    fraud = int(df["Class"].sum())
    normal = total-fraud

    fraud_percent = fraud/total*100
    normal_percent = normal/total*100

    d1,d2,d3,d4 = st.columns(4)

    d1.metric("💳 Total",total)
    d2.metric("✅ Legitimate",normal)
    d3.metric("🚨 Fraud",fraud)
    d4.metric("📌 Fraud %",f"{fraud_percent:.3f}%")

    st.divider()

    # -----------------------
    # DISTRIBUTION
    # -----------------------

    st.subheader("📉 Transaction Distribution")

    chart = pd.DataFrame({
        "Type":["Legitimate","Fraud"],
        "Transactions":[normal,fraud]
    })

    fig = px.bar(
        chart,
        x="Type",
        y="Transactions",
        color="Type",
        text="Transactions",
        title="Credit Card Transaction Distribution"
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # -----------------------
    # PIE CHART
    # -----------------------

    st.subheader("🥧 Fraud vs Legitimate")

    pie = px.pie(
        chart,
        values="Transactions",
        names="Type",
        hole=0.5
    )

    st.plotly_chart(pie,use_container_width=True)

    st.divider()

    # -----------------------
    # CONFUSION MATRIX
    # -----------------------

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test,y_pred)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Normal","Predicted Fraud"],
            y=["Actual Normal","Actual Fraud"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    fig_cm.update_layout(height=450)

    st.plotly_chart(fig_cm,use_container_width=True)

    st.divider()

    # -----------------------
    # FEATURE IMPORTANCE
    # -----------------------

    st.subheader("⭐ Top 10 Important Features")

    importance = pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig2 = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Most Influential Features"
    )

    fig2.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # -----------------------
    # ROC CURVE
    # -----------------------

    st.subheader("📈 ROC Curve")

    from sklearn.metrics import roc_curve

    fpr,tpr,_ = roc_curve(y_test,y_prob)

    roc = pd.DataFrame({
        "False Positive Rate":fpr,
        "True Positive Rate":tpr
    })

    fig3 = px.line(
        roc,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {auc:.4f})"
    )

    fig3.add_scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        name="Random Guess"
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.divider()

    # -----------------------
    # BUSINESS INSIGHTS
    # -----------------------

    st.subheader("💼 Business Insights")

    st.success(f"""
✔ The model correctly identifies fraudulent transactions with an **Accuracy of {acc:.4f}**.

✔ ROC-AUC Score of **{auc:.4f}** indicates excellent discrimination between fraud and legitimate transactions.

✔ Only **{fraud_percent:.3f}%** of all transactions are fraudulent, confirming a highly imbalanced dataset.

✔ SMOTE was applied during model training to improve fraud detection performance.
""")

# =====================================================
# DATASET
# =====================================================

elif page=="Dataset":

    st.subheader("Dataset")

    st.dataframe(df.head(20),use_container_width=True)

    st.write("Shape :",df.shape)

    st.write("Missing Values")

    st.write(df.isnull().sum())

    st.write("Statistics")

    st.dataframe(df.describe())

# =====================================================
# PREDICTION
# =====================================================

# =====================================================
# PREDICTION
# =====================================================

else:

    st.header("🔍 Transaction Prediction")

    st.info(
        "Enter transaction details below to predict whether a transaction is Legitimate or Fraudulent."
    )

    cols = df.drop("Class", axis=1).columns

    values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(cols):

        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )
        else:
            with col2:
                value = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )

        values.append(value)

    st.divider()

    if st.button("🚀 Predict Transaction", use_container_width=True):

        input_df = pd.DataFrame([values], columns=cols)

        scaled = scaler.transform(input_df)

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0][1]

        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
        )

        c2.metric(
            "Prediction",
            "Fraud" if prediction == 1 else "Legitimate"
        )

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    # ===========================================
    # Batch Prediction
    # ===========================================

    st.divider()

    st.header("📂 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(batch_df.head())

        try:

            scaled_data = scaler.transform(batch_df)

            predictions = model.predict(scaled_data)

            probabilities = model.predict_proba(scaled_data)[:, 1]

            batch_df["Prediction"] = [
                "Fraud" if p == 1 else "Legitimate"
                for p in predictions
            ]

            batch_df["Fraud Probability"] = probabilities

            st.subheader("Prediction Results")

            st.dataframe(batch_df)

            csv = batch_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results",
                csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"❌ {e}")