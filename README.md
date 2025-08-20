# Customer Segmentation for E-Commerce

## Overview
This project demonstrates a structured, reproducible data science workflow for customer segmentation in the e-commerce domain.  
The workflow integrates:

- **RFM Analysis**   
- **Clustering** (K-Means)  
- **Classification** (KNN)  
- **Interactive Dashboard** (Streamlit)  

The *Online Retail II* dataset (Chen, Sain & Guo, 2012) was used for this Project. 

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/MirasWalid/Customer-Segmentation-for-E-Commerce.git
cd Customer-Segmentation-for-E-Commerce
pip install -r requirements.txt
```

---

## Launch Dashboard
```bash
streamlit run rfm_dashboard.py
```

## Project Structure
```
├── app/
│   └── rfm_dashboard.py               # Streamlit dashboard
├── data/
│   ├── online_retail_II.xlsx          # Raw dataset
│   ├── rfm_features_transformed.csv   # Processed RFM data
│   ├── rfm_clusters.csv               # Clustered data
│   └── classification_report.json     # Model evaluation metrics
├── notebooks/
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── classification-checkpoint.py
├── requirements.txt
└── README.md
```



