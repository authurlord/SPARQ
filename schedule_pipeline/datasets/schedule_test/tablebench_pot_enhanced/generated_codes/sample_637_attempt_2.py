import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'Applications' and 'Offer Rate (%)' to numeric
df['Applications'] = df['Applications'].str.replace(',', '').astype(float)
df['Offer Rate (%)'] = pd.to_numeric(df['Offer Rate (%)'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df[['Applications', 'Offer Rate (%)']].dropna()

# Calculate correlation coefficient
correlation = df_clean['Applications'].corr(df_clean['Offer Rate (%)'])

print(f"Final Answer: {correlation:.4f}")