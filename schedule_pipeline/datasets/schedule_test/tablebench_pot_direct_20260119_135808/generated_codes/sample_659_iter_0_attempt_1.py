import pandas as pd

df = pd.read_csv('table.csv')
# Clean 'death toll' by removing commas and converting to integer
df['death toll'] = df['death toll'].str.replace(',', '').astype(int)
# Convert 'magnitude' to float
df['Magnitude'] = df['Magnitude'].astype(float)
# Calculate correlation coefficient
correlation = df['Magnitude'].corr(df['death toll'])
print(f"Final Answer: {correlation:.3f}")