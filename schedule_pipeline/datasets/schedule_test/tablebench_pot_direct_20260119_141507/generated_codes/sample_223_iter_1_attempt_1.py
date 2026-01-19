import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'mccain %' column from string with '%' to float
df['mccain %'] = df['mccain %'].str.replace('%', '').astype(float)

# Calculate the correlation between 'mccain %' and 'total'
correlation = df['mccain %'].corr(df['total'])

print(f"Final Answer: {correlation:.3f}")