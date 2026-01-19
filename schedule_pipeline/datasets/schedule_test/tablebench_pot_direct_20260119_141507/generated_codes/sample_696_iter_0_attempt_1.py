import pandas as pd

df = pd.read_csv('table.csv')

# Convert the 'foreign' column to numeric (some entries may be strings)
df['foreign'] = pd.to_numeric(df['foreign'], errors='coerce')

# Find the maximum and minimum values in the 'foreign' column
max_foreign = df['foreign'].max()
min_foreign = df['foreign'].min()

# Find the municipality with the highest and lowest values
max_municipality = df.loc[df['foreign'].idxmax(), df.columns[0]]  # This is not correct — we need to find which column has max/min
# Actually, we need to identify the municipality name (column) that has max/min

# Correct approach: iterate through columns to find max and min per column, then find the municipality with max and min
foreign_values = df['foreign'].values
municipalities = df.columns[1:]  # Skip 'language'

# Create a dictionary of values per municipality
municipality_foreign = {}
for col in municipalities:
    municipality_foreign[col] = df[col].astype(float).fillna(0)

# Find the municipality with highest and lowest foreign language speakers
highest_mun = max(municipality_foreign, key=lambda x: municipality_foreign[x])
lowest_mun = min(municipality_foreign, key=lambda x: municipality_foreign[x])

difference = municipality_foreign[highest_mun] - municipality_foreign[lowest_mun]

print(f"Final Answer: {highest_mun}, {difference:.1f}")