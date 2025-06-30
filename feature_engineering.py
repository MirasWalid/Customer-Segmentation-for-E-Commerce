# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
data = pd.read_csv('data/online_retail_II.csv', encoding='ISO-8859-1')

# Check columns
print(data.columns)

# Remove rows with missing CustomerID
data = data[pd.notnull(data['Customer ID'])]

# Remove canceled orders (Invoice numbers that start with 'C')
data = data[~data['Invoice'].astype(str).str.startswith('C')]

# Create 'TotalPrice' column
data['TotalPrice'] = data['Quantity'] * data['Price']

# Define reference date (day after last invoice)
reference_date = data['InvoiceDate'].max()
reference_date = pd.to_datetime(reference_date) + pd.Timedelta(days=1)

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Calculate RFM metrics per customer
rfm = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                       # Monetary
})

#  Rename columns
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Optional: Remove customers with negative or zero Monetary
rfm = rfm[rfm['Monetary'] > 0]

# Save RFM table (optional)
rfm.to_csv('data/rfm_features.csv')

# Check result
print(rfm.describe())
rfm.head()
