import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers' column to numeric, removing any non-numeric characters
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['viewers'].quantile(0.25)
Q3 = df['viewers'].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify anomalies (outliers)
anomalies = df[(df['viewers'] < lower_bound) | (df['viewers'] > upper_bound)]

# Extract titles of episodes with anomalies
anomaly_titles = anomalies['title'].tolist()

print(f"Final Answer: {', '.join(anomaly_titles)}")