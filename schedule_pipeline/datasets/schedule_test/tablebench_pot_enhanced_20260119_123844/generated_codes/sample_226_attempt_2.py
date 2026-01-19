import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
df['live births per year'] = df['live births per year'].str.replace(' ', '').astype(int)
df['deaths per year'] = df['deaths per year'].str.replace(' ', '').astype(int)

# Calculate total live births and total deaths over all periods
total_births = df['live births per year'].sum()
total_deaths = df['deaths per year'].sum()

# Determine which is the primary driver
if total_births > total_deaths:
    primary_driver = "live births per year"
else:
    primary_driver = "deaths per year"

print(f"Final Answer: {primary_driver}")