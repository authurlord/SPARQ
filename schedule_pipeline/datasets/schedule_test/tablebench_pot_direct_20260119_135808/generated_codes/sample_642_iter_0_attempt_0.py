import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'starting weight (kg)' to numeric
df['starting weight (kg)'] = pd.to_numeric(df['starting weight (kg)'])
# Convert 'percentage lost' to numeric, removing '%' sign
df['percentage lost'] = pd.to_numeric(df['percentage lost'].str.replace('%', ''))
# Calculate correlation coefficient
correlation = df['starting weight (kg)'].corr(df['percentage lost'])
print(f"Final Answer: {correlation:.4f}")