import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Applications' to numeric, removing commas
df['Applications'] = df['Applications'].str.replace(',', '').astype(float)

# Convert 'Offer Rate (%)' to numeric
df['Offer Rate (%)'] = pd.to_numeric(df['Offer Rate (%)'])

# Extract data for 2013 to 2017
applications = df['Applications']
offer_rate = df['Offer Rate (%)']

# Calculate correlation coefficient
correlation = applications.corr(offer_rate)

print(f"Final Answer: {correlation:.3f}")