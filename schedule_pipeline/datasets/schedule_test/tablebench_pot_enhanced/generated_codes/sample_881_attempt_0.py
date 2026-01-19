import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains header descriptions
df = df.drop(df.index[-1])
# Set the first column as index (years)
df = df.set_index('-')
# Convert all columns to numeric, excluding the first column
df = df.apply(pd.to_numeric, errors='coerce')

# Get the 1939 data
year_1939 = df.loc['1939']
max_import_country = year_1939.idxmax()
max_import_1939 = year_1939.max()

# Get the 1934 value for the same country
max_import_1934 = df.loc['1934'][max_import_country]

# Compare the values
comparison = max_import_1939 - max_import_1934

print(f"Final Answer: {max_import_country}, {comparison:.1f}")