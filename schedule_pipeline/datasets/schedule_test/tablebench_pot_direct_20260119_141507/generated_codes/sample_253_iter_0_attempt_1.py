import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'PSI' column to numeric, replacing '-' with NaN and then dropping NaN
df['PSI'] = pd.to_numeric(df['PSI'], errors='coerce')
# Remove any rows where PSI is NaN
df = df.dropna(subset=['PSI'])
# Find max and min PSI values
max_psi = df['PSI'].max()
min_psi = df['PSI'].min()
difference = max_psi - min_psi
print(f"Final Answer: {difference:.1f}")