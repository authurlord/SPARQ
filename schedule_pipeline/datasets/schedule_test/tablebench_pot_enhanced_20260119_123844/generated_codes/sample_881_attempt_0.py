import pandas as pd

df = pd.read_csv('table.csv')
# Remove the last row which contains the header description
df = df.iloc[:-1]
# Set the first column as index (years)
df.set_index('-', inplace=True)
# Convert columns to numeric
df = df.apply(pd.to_numeric)

# Find the country with highest imports in 1939
max_1939_country = df.loc['1939'].idxmax()
max_1939_value = df.loc['1939', max_1939_country]
max_1934_value = df.loc['1934', max_1939_country]

print(f"Final Answer: {max_1939_country}, {max_1934_value}")