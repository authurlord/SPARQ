import pandas as pd

df = pd.read_csv('table.csv')

# Convert local magnitude to float
df['local magnitude'] = pd.to_numeric(df['local magnitude'], errors='coerce')

# Compute IQR-based outliers
Q1 = df['local magnitude'].quantile(0.25)
Q3 = df['local magnitude'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['local magnitude'] < lower_bound) | (df['local magnitude'] > upper_bound)]

# Display the rows with abnormal data points
print(outliers[['date', 'local magnitude', 'location']])
Final Answer: 3 may 1949, 10.5, yerecoin