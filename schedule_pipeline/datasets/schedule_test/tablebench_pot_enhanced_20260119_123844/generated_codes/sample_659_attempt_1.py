import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'death toll' to numeric by removing commas
df['death toll'] = df['death toll'].str.replace(',', '').astype(int)
# Convert 'magnitude' to float
df['Magnitude'] = df['Magnitude'].astype(float)
# Calculate correlation coefficient
correlation = df['Magnitude'].corr(df['death toll'])
print(f"Final Answer: {correlation:.4f}")