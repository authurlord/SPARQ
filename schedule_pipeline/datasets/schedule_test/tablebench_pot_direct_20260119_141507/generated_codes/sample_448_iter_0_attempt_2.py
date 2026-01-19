import pandas as pd

df = pd.read_csv('table.csv')

# Convert viewers to numeric (some entries might be strings)
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Remove any NaN values
df = df.dropna(subset=['viewers'])

# Calculate Q1, Q3, and IQR
Q1 = df['viewers'].quantile(0.25)
Q3 = df['viewers'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find anomalies (values outside bounds)
anomalies = df[(df['viewers'] < lower_bound) | (df['viewers'] > upper_bound)]

# Extract the viewership values that are anomalies
anomaly_viewers = anomalies['viewers'].tolist()

print(f"Final Answer: {anomaly_viewers}")