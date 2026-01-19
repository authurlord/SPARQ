import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'viewers (m)' and 'rating' to numeric, dropping rows with 'tba'
df['viewers (m)'] = pd.to_numeric(df['viewers (m)'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows where either column is NaN (i.e., 'tba')
df_clean = df.dropna(subset=['viewers (m)', 'rating'])

# Calculate correlation
correlation = df_clean['viewers (m)'].corr(df_clean['rating'])

print(f"Final Answer: {correlation:.2f}")