import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains the header description
df = df.drop(df.index[-1])
# Convert columns to numeric, excluding the first column (year)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

# Filter data for 1939
data_1939 = df[df['-'] == '1939']
# Find the country with the highest imports in 1939
max_country_1939 = data_1939.iloc[0][1:].idxmax()
max_value_1939 = data_1939.iloc[0][max_country_1939]

# Get the 1934 value for that country
data_1934 = df[df['-'] == '1934']
max_value_1934 = data_1934.iloc[0][max_country_1939]

print(f"Final Answer: {max_country_1939}, {max_value_1934}")