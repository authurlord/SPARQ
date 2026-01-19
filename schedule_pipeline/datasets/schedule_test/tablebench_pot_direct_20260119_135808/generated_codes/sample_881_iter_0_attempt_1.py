import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains the header description
df = df.iloc[:-1]
# Set the first column as index (years)
df.set_index('-', inplace=True)
# Extract 1939 and 1934 data
year_1939 = df.loc['1939'].astype(float)
year_1934 = df.loc['1934'].astype(float)

# Find the country with highest imports in 1939
max_country_1939 = year_1939.idxmax()
value_1939 = year_1939.max()
value_1934 = year_1934[year_1939.idxmax()]

print(f"Final Answer: {max_country_1939}, {value_1939:.1f}, {value_1934:.1f}")