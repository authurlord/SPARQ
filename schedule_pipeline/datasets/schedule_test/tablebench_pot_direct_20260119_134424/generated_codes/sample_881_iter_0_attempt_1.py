import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains the header description
df = df.iloc[:-1]
# Convert columns to numeric, excluding the first column (year)
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Filter for 1939
year_1939 = df[df['-'] == '1939']
# Find the country with the highest imports in 1939
max_country_1939 = year_1939.iloc[0].idxmax()
max_value_1939 = year_1939[max_country_1939].values[0]

# Get the 1934 value for the same country
year_1934 = df[df['-'] == '1934']
value_1934 = year_1934[max_country_1939].values[0]

print(f"Final Answer: {max_country_1939}, {max_value_1939}, {value_1934}")