import pandas as pd

df = pd.read_csv('table.csv')
# Drop the last row which contains the header description
df = df.iloc[:-1]
# Convert columns to numeric, excluding the first column (year)
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Filter data for 1939
data_1939 = df[df['-'] == '1939']
max_import_country = data_1939.iloc[0].idxmax()
max_import_1939 = data_1939[max_import_country].values[0]

# Get 1934 import for the same country
data_1934 = df[df['-'] == '1934']
max_import_1934 = data_1934[max_import_country].values[0]

print(f"Final Answer: {max_import_country}, {max_import_1939 - max_import_1934:.1f}")