import pandas as pd

df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric, handling commas and percentages
df['Applications'] = df['Applications'].str.replace(',', '').astype(int)
df['Offer Rate (%)'] = df['Offer Rate (%)'].str.replace('%', '').astype(float)

# Select the years 2013 to 2017 (rows corresponding to the data in the table)
# The data is already aligned by year in the columns, so we extract the values directly
applications = df['Applications'].values
offer_rate = df['Offer Rate (%)'].values

# Compute the correlation coefficient
correlation_coefficient = df[['Applications', 'Offer Rate (%)']].corr().iloc[0, 1]

print(f"Final Answer: {correlation_coefficient:.3f}")