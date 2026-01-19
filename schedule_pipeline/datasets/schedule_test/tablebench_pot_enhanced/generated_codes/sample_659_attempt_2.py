import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'death toll' by removing commas and converting to integer
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(int)
# Convert 'Magnitude' to float
df['Magnitude'] = df['Magnitude'].astype(float)
# Calculate correlation coefficient
correlation = df['Magnitude'].corr(df['Death toll'])
print(f"Final Answer: {correlation:.4f}")