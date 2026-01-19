import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to numeric, handling any parsing issues
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna(subset=['viewers'])

# Extract viewership data
viewers = df['viewers']

# Calculate Q1, Q3, and IQR
Q1 = viewers.quantile(0.25)
Q3 = viewers.quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find anomalies (outliers)
anomalies = viewers[(viewers < lower_bound) | (viewers > upper_bound)]

print(f"Final Answer: {anomalies.tolist()}")