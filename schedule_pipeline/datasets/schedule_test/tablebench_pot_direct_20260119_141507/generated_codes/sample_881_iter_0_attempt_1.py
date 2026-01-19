import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the last row which is a header comment
df = df.iloc[:-1]

# Convert the columns to numeric (remove any non-numeric issues)
df = df.apply(pd.to_numeric, errors='coerce')

# Extract values for 1934 and 1939
year_1934 = df.iloc[0]  # First row after header
year_1939 = df.iloc[5]  # Sixth row (index 5) after header

# Find the country with the highest import in 1939
max_1939_value = year_1939.max()
max_1939_country = year_1939.idxmax()

# Get the 1934 value for that country
value_1934 = year_1934[max_1939_country]

# Compare: difference or ratio
difference = value_1939[max_1939_country] - value_1934

print(f"Final Answer: {max_1939_country}, {difference:.1f}")