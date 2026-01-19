import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows where 'rating' is 'tba'
df_clean = df[df['rating'] != 'tba']

# Convert 'viewers (m)' and 'rating' to numeric
df_clean['viewers (m)'] = pd.to_numeric(df_clean['viewers (m)'], errors='coerce')
df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')

# Remove any rows with NaN after conversion
df_clean = df_clean.dropna()

# Calculate the correlation between viewers (m) and rating
correlation = df_clean['viewers (m)'].corr(df_clean['rating'])

print(f"Final Answer: {correlation:.2f}")