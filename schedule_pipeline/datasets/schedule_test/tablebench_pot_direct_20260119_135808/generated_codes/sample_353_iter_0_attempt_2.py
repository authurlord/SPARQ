import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'per capita income' to numeric, removing any non-numeric characters if needed
df['per capita income'] = pd.to_numeric(df['per capita income'], errors='coerce')

# Filter counties with per capita income between 18,000 and 27,000 (inclusive)
filtered_counties = df[(df['per capita income'] >= 18000) & (df['per capita income'] <= 27000)]

# Count the number of such counties
count = len(filtered_counties)
print(f"Final Answer: {count}")