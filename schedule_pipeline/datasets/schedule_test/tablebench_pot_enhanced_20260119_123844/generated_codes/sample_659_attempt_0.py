import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'death toll' column by removing commas and converting to float
df['death toll'] = df['Death toll'].str.replace(',', '').astype(float)
# Convert 'magnitude' to float
df['magnitude'] = df['Magnitude'].astype(float)
# Calculate correlation coefficient
correlation = df['magnitude'].corr(df['death toll'])
print(f"Final Answer: {correlation:.3f}")