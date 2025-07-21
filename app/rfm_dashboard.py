import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import io
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
import plotly.express as px
import base64

# Page configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("../data/rfm_clusters.csv", index_col=0)

df = load_data()

# Sidebar filters
st.sidebar.header("Filter")
selected_clusters = st.sidebar.multiselect(
    "Select clusters", sorted(df["Cluster"].unique()), default=df["Cluster"].unique()
)

df_filtered = df[df["Cluster"].isin(selected_clusters)]

# Main title
st.title("Customer Segmentation & Classification Dashboard")

# Define dashboard tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Key Metrics",
    "2. Data Summary",
    "3. Visualizations",
    "4. Segmentation",
    "5. Model Insights",
    "6. Recommendations",
    "7. Export"
])

# Tab 1: Key Metrics
with tab1:
    st.header("Key Metrics")

    try:
        with open("../data/classification_report.json", "r") as f:
            metrics = json.load(f)

        accuracy = metrics.get("accuracy", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1-score", 0)

    except FileNotFoundError:
        accuracy = precision = recall = f1 = 0
        st.warning("classification_report.json not found.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    # Simulated business metrics
    col5, col6, col7 = st.columns(3)
    col5.metric("Revenue (simulated)", f"{df_filtered['Monetary Value_log'].sum():.2f}")
    col6.metric("Number of Customers", f"{df_filtered.shape[0]}")
    col7.metric("Estimated Churn Rate", "17%")

# Tab 2: Data Summary
with tab2:
    st.header("Data Summary")
    st.write("Shape:", df_filtered.shape)
    st.write("Features:", list(df_filtered.columns))
    st.dataframe(df_filtered.head())

# Tab 3: Visualizations
with tab3:
    st.header("Visualizations")

    st.subheader("Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df_filtered, x="Cluster", ax=ax)
    st.pyplot(fig)

    st.subheader("RFM Distributions (Histograms)")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_filtered["Recency"], kde=True, ax=axs[0])
    axs[0].set_title("Recency")
    sns.histplot(df_filtered["Frequency_log"], kde=True, ax=axs[1])
    axs[1].set_title("Frequency (log)")
    sns.histplot(df_filtered["Monetary Value_log"], kde=True, ax=axs[2])
    axs[2].set_title("Monetary Value (log)")
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Scatter Plot")
    fig = px.scatter(
        df_filtered,
        x="Recency",
        y="Monetary Value_log",
        color="Cluster",
        hover_data=df_filtered.columns
    )
    st.plotly_chart(fig)

# Tab 4: Segmentation
with tab4:
    st.header("Segmentation Analysis")

    st.write("Average RFM values per cluster:")
    st.dataframe(df_filtered.groupby("Cluster")[["Recency", "Frequency_log", "Monetary Value_log"]].mean())

    st.subheader("Boxplots per Cluster")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=df_filtered, x="Cluster", y="Recency", ax=axs[0])
    sns.boxplot(data=df_filtered, x="Cluster", y="Frequency_log", ax=axs[1])
    sns.boxplot(data=df_filtered, x="Cluster", y="Monetary Value_log", ax=axs[2])
    st.pyplot(fig)

# Tab 5: Model Insights
with tab5:
    st.header("Model Insights")

    st.subheader("Confusion Matrix (example)")
    cm = confusion_matrix([0, 1, 2, 1, 0], [0, 2, 1, 1, 0])  # example only
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve (example)")
    fpr = [0, 0.1, 0.3, 1]
    tpr = [0, 0.6, 0.85, 1]
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# Tab 6: Recommendations
with tab6:
    st.header("Recommendations")

    st.markdown("**Cluster-specific suggestions:**")
    st.write("- Cluster 0: Low frequency customers – consider reactivation email campaigns.")
    st.write("- Cluster 1: High value, moderate recency – offer loyalty incentives.")
    st.write("- Cluster 2: New customers – provide onboarding support and welcome offers.")

    st.markdown("**Warnings:**")
    st.warning("High Recency in Cluster 0 – potential churn risk.")

# Tab 7: Export Options
with tab7:
    st.header("Export Options")

    csv = df_filtered.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="rfm_export.csv">Download CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)
