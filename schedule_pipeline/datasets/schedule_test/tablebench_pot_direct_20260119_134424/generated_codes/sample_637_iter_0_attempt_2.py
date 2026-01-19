import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert data
df['Applications'] = df['Applications'].str.replace(',', '').astype(float)
df['Offer Rate (%)'] = df['Offer Rate (%)'].astype(float)

# Calculate correlation coefficient
correlation = df['Applications'].corr(df['Offer Rate (%)'])

print(f"Final Answer: {correlation:.3f}")