import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'PSI' column to numeric, replacing '-' with NaN
df['PSI'] = pd.to_numeric(df['PSI'], errors='coerce')
# Drop rows where PSI is NaN
df.dropna(subset=['PSI'], inplace=True)
# Calculate the difference between max and min PSI
psi_difference = df['PSI'].max() - df['PSI'].min()
print(f"Final Answer: {psi_difference:.1f}")