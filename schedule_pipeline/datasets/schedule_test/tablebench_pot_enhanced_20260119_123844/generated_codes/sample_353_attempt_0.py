import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'per capita income' to numeric, handling any non-numeric values
df['per capita income'] = pd.to_numeric(df['per capita income'], errors='coerce')
# Filter counties with per capita income between 18000 and 27000
filtered_counties = df[(df['per capita income'] >= 18000) & (df['per capita income'] <= 27000)]
# Count the number of such counties
count = len(filtered_counties)
print(f"Final Answer: {count}")