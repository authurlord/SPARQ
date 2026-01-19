import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'Applications' column: remove commas and convert to numeric
df['Applications'] = df['Applications'].str.replace(',', '').astype(float)

# Convert 'Offer Rate (%)' to numeric
df['Offer Rate (%)'] = pd.to_numeric(df['Offer Rate (%)'])

# Extract data for 2013 to 2017 (columns 2013 to 2017)
applications = df['Applications']
offer_rate = df['Offer Rate (%)']

# Calculate correlation coefficient
correlation_coefficient = applications.corr(offer_rate)

print(f"Final Answer: {correlation_coefficient:.3f}")