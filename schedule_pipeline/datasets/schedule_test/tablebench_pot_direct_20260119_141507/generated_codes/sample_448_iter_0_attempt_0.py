import pandas as pd

df = pd.read_csv('table.csv')

# Convert viewers to numeric (some might be strings)
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Remove any NaN values
df = df.dropna(subset=['viewers'])

# Calculate Q1, Q3, and IQR
Q1 = df['viewers'].quantile(0.25)
Q3 = df['viewers'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find anomalies
anomalies = df[(df['viewers'] < lower_bound) | (df['viewers'] > upper_bound)]

# Extract the titles of episodes with anomalies
anomaly_titles = anomalies['title'].tolist()

print(f"Final Answer: {', '.join(anomaly_titles)}")