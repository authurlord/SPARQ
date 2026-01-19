import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'PSI' column to numeric, treating '-' as NaN
df['PSI'] = pd.to_numeric(df['PSI'], errors='coerce')
# Drop rows where PSI is NaN
df_clean = df.dropna(subset=['PSI'])
# Calculate the difference between max and min PSI
difference = df_clean['PSI'].max() - df_clean['PSI'].min()
print(f"Final Answer: {difference:.1f}")