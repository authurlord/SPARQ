import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'death toll' by removing commas and converting to numeric
df['death toll'] = df['Death toll'].str.replace(',', '').astype(float)
# Convert 'magnitude' to float
df['Magnitude'] = df['Magnitude'].astype(float)
# Calculate correlation coefficient
correlation = df['Magnitude'].corr(df['death toll'])
print(f"Final Answer: {correlation:.4f}")