import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric, coercing errors to NaN
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with missing values (e.g., the '191822' row)
df_clean = df.dropna()

# Calculate correlation between typhus and typhoid fever
correlation = df_clean['typhus'].corr(df_clean['typhoid fever'])

print(f"Final Answer: {correlation:.2f}")