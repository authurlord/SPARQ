import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains headers
df = df.iloc[:-1]
# Set the first column as index
df.set_index('-', inplace=True)
# Convert data to float
df = df.astype(float)

# Find the country with highest imports in 1939
max_1939_country = df.loc['1939'].idxmax()
max_1939_value = df.loc['1939', max_1939_country]
max_1934_value = df.loc['1934', max_1939_country]

print(f"Final Answer: {max_1939_country}, {max_1934_value}")