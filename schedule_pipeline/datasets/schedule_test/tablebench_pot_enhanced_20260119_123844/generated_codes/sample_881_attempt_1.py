import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains headers
df = df.iloc[:-1]
# Set the first column as index (year)
df.set_index('-', inplace=True)
# Convert all columns to numeric
df = df.astype(float)

# Find the country with highest imports in 1939
max_1939 = df.loc['1939'].idxmax()
value_1939 = df.loc['1939', max_1939]
value_1934 = df.loc['1934', max_1939]

print(f"Final Answer: {max_1939}, {value_1939:.1f}, {value_1934:.1f}")