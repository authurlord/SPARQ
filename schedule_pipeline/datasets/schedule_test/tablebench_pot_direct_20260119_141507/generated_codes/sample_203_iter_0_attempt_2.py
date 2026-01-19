import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'viewers (m)' and 'rating' to numeric, handling errors
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows with missing values
df_clean = df.dropna(subset=['viewers (m)', 'rating'])

# Calculate the correlation between viewers (m) and rating
correlation = df_clean['viewers (m)'].corr(df_clean['rating'])

print(f"Final Answer: {correlation:.2f}")