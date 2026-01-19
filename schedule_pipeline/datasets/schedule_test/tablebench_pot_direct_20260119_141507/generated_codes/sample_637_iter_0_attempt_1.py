import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric (remove commas and handle non-numeric entries)
applications = pd.to_numeric(df['Applications'].str.replace(',', ''), errors='coerce')
offer_rate = pd.to_numeric(df['Offer Rate (%)'], errors='coerce')

# Drop any NaN values due to conversion errors
applications = applications.dropna()
offer_rate = offer_rate.dropna()

# Compute correlation coefficient
correlation_coefficient = np.corrcoef(applications, offer_rate)[0, 1]

print(f"Final Answer: {correlation_coefficient:.3f}")