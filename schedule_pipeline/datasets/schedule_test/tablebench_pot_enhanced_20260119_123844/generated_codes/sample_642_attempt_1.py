import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'starting weight (kg)' to numeric
df['starting weight (kg)'] = pd.to_numeric(df['starting weight (kg)'])
# Convert 'percentage lost' to numeric by removing '%' and converting to float
df['percentage lost'] = df['percentage lost'].str.replace('%', '').astype(float)
# Calculate correlation between 'starting weight (kg)' and 'percentage lost'
correlation = df['starting weight (kg)'].corr(df['percentage lost'])
print(f"Final Answer: {correlation:.4f}")