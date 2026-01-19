import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Applications' column to numeric, removing commas
df['Applications'] = df['Applications'].str.replace(',', '').astype(float)
# Convert 'Offer Rate (%)' column to numeric
df['Offer Rate (%)'] = df['Offer Rate (%)'].astype(float)
# Calculate correlation coefficient
correlation = df['Applications'].corr(df['Offer Rate (%)'])
print(f"Final Answer: {correlation:.3f}")