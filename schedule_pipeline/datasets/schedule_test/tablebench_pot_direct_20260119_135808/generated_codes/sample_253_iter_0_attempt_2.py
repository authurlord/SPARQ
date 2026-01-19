import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'PSI' column to numeric, coercing errors (like '-') to NaN
df['PSI'] = pd.to_numeric(df['PSI'], errors='coerce')
# Drop rows where PSI is NaN (due to '-')
df = df.dropna(subset=['PSI'])
# Calculate the difference between max and min PSI
psi_difference = df['PSI'].max() - df['PSI'].min()
print(f"Final Answer: {psi_difference:.1f}")