import pandas as pd

df = pd.read_csv('table.csv')

# Drop the last row (header comment)
df = df.iloc[:-1]

# Convert all values in the data columns to float
df = df.apply(pd.to_numeric, errors='coerce')

# Extract 1934 and 1939 values
year_1934 = df.iloc[0]
year_1939 = df.iloc[5]

# Find the country with the highest value in 1939
max_1939_country = year_1939.idxmax()
value_1939 = year_1939[max_1939_country]
value_1934 = year_1934[max_1939_country]
difference = value_1939 - value_1934

print(f"Final Answer: {max_1939_country}, {difference:.1f}")