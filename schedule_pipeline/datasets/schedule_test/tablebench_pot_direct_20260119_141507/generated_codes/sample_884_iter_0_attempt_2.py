import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the data to a DataFrame
df = df.dropna()

# Convert relevant columns to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'], errors='coerce')
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'], errors='coerce')
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'], errors='coerce')

# Sort by year
df = df.sort_values('year').reset_index(drop=True)

# Calculate year-over-year percentage increases
df['domestic_increase'] = ((df['domestic s ton'] - df['domestic s ton'].shift(1)) / df['domestic s ton'].shift(1)) * 100
df['foreign_increase'] = ((df['foreign total s ton'] - df['foreign total s ton'].shift(1)) / df['foreign total s ton'].shift(1)) * 100

# Find the year where domestic increase > foreign increase
result_year = df[df['domestic_increase'] > df['foreign_increase']]['year'].tolist()

# If no such year, return None, otherwise return the first (or all) such year
if result_year:
    Final Answer: result_year[0]
else:
    Final Answer: "No such year"